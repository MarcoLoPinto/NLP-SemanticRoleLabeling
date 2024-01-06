'''
    The Net part
'''

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoConfig

class Net_aic_transformer_lstm(nn.Module):
    """ argument identification + argument classification model """
    def __init__(self, fine_tune_transformer = False, hparams = {}, loss_fn = None, load_transformer_config = False):
        """argument identification + argument classification model

        Args:
            fine_tune_transformer (bool, optional): if the transformer weights needs to be fine-tuned or freezed. Defaults to False.
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
            load_transformer_config (bool, optional): if the model needs to be entirely loaded or just the configuration (used to speed-up the loading process in the evaluation part). Defaults to False.
        """
        super().__init__()

        hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.n_labels = int(hparams['n_roles_labels'])
        self.n_predicates = int(hparams['n_predicates_labels'])
        predicates_mask_dim = 1
        
        # 1) Embedding
        if load_transformer_config:
            config = AutoConfig.from_pretrained(hparams['transformer_name'])
            self.transformer_model = AutoModel.from_config(config)
            self.transformer_model.config.output_hidden_states = True
        else:
            self.transformer_model = AutoModel.from_pretrained(
                hparams['transformer_name'], output_hidden_states=True
            )
        if not fine_tune_transformer:
            for param in self.transformer_model.parameters():
                param.requires_grad = fine_tune_transformer

        transformer_out_dim = self.transformer_model.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        # 2) Sequence encoder

        word_lstm_n_layers = 1
        word_lstm_hidden_size = 256
        word_lstm_bidirectional = True
        self.word_lstm = nn.LSTM(
            input_size = transformer_out_dim + self.n_predicates + predicates_mask_dim, 
            hidden_size = word_lstm_hidden_size, 

            bidirectional = word_lstm_bidirectional, 
            num_layers = word_lstm_n_layers, 
            dropout = 0.3 if word_lstm_n_layers > 1 else 0,
            batch_first = True 
        )
        word_lstm_dim_out = (2*word_lstm_hidden_size if word_lstm_bidirectional is True else word_lstm_hidden_size)

        # 3) Classifier

        self.fc1 = nn.Linear(word_lstm_dim_out, word_lstm_dim_out // 2)
        self.ln1 = nn.LayerNorm(word_lstm_dim_out // 2)
        self.relu = nn.ReLU()

        self.classifier = nn.Linear(word_lstm_dim_out // 2, self.n_labels)

        # Loss function:
        self.loss_fn = loss_fn
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        batch_sentence_predicates, 
        batch_predicate_position,
        token_type_ids = None,
    ):
        """_summary_

        Args:
            input_ids (torch.Tensor): the tensors inputs generated via the Tokenizer
            attention_mask (torch.Tensor): attention mask generated via the Tokenizer
            batch_sentence_predicates (torch.Tensor): it has the predicates labels for each word (in the training/test, there is only one for sentence!). It has shape (batch, sentence_len)
            batch_predicate_position (torch.Tensor): the position in the sentence of the target predicate. It has shape (batch)
            token_type_ids (any, optional): generated via the Tokenizer. Defaults to None.

        Returns:
            torch.Tensor: the logits
        """

        # 1) Embedding

        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        # some transformer models don't have token_type_ids parameter... checking 
        if token_type_ids is None: 
            transformer_kwargs['token_type_ids'] = token_type_ids
        # number of hidden states to consider from the transformer
        n_transformer_hidden_states = 4
        
        transformer_outs = self.transformer_model(**transformer_kwargs)
        # summing all the considered dimensions
        transformer_out = torch.stack(
            transformer_outs.hidden_states[-n_transformer_hidden_states:],
            dim=0).sum(dim=0)

        batch_sentence_words = self.dropout(transformer_out) # -> (batch, sentence_len, word_emb_dim)

        # 2) Sequence encoder

        # For each sentence in the batch, it generates an array of the same length as the sentence:
        # the only 1 in each array corresponds to the position of the predicate in the sentence.
        # If the position of the target predicate is equal to -1, then the encoding for it becomes [0,0,...,0] 
        # (i.e. there is no target predicate for that sentence).
        # predicate_mask has shape (batch, sentence_len)
        predicate_mask = torch.nn.functional.one_hot( 
            torch.where(batch_predicate_position >=0 , batch_predicate_position , 0) , 
            num_classes=input_ids.shape[-1] 
        ) * (batch_predicate_position >= 0)[:,None]

        # For each word in the sentence (or not, like '_') it associates a one-hot representation
        # for that particular word.
        # If a predicate is equal to -1, then the encoding for it becomes [0,0,...,0]
        # (i.e. that particular word in the sentence is padding, so convert to a vector of zeros)
        # batch_sentence_predicates_onehot has shape (batch, sentence_len, n_predicates)
        batch_sentence_predicates_onehot = torch.nn.functional.one_hot( 
            torch.where(batch_sentence_predicates >=0 , batch_sentence_predicates , 0), 
            num_classes=self.n_predicates,
        ) * (batch_sentence_predicates >= 0)[:,:,None]
    
        batch_sentence_words = torch.cat((
            batch_sentence_words, 
            batch_sentence_predicates_onehot, 
            predicate_mask.unsqueeze(-1),
        ), dim=-1)

        batch_sentence_words, _ = self.word_lstm(batch_sentence_words)

        # 3) Classifier

        batch_sentence_words = self.fc1(batch_sentence_words)
        batch_sentence_words = self.ln1(batch_sentence_words)
        batch_sentence_words = self.relu(batch_sentence_words)
        
        logits = self.classifier(batch_sentence_words)

        return logits # (batch, sentence_len, n_lables)

    def compute_loss(self, x, y_true):
        """computes the loss for the net

        Args:
            x (torch.Tensor): The predictions
            y_true (torch.Tensor): The true labels

        Returns:
            any: the loss
        """
        if self.loss_fn is None:
            return None
        return self.loss_fn(x, y_true)

    def get_indices(self, torch_outputs):
        """
        Args:
            torch_outputs (Tensor): a Tensor with shape (batch_size, sentence_len, label_vocab_size) containing the logits outputed by the neural network (if batch_first = True)
        Output:
            The method returns a (batch_size, sentence_len) shaped tensor (if batch_first = True)
        """
        max_indices = torch.argmax(torch_outputs, -1) # resulting shape = (batch_size, sentence_len)
        return max_indices
    
    def load_weights(self, path, strict = True):
        """load the weights of the model

        Args:
            path (str): path to the saved weights
            strict (bool, optional): Strict parameter for the torch.load() function. Defaults to True.
        """
        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device), strict=strict)
        self.eval()
    
    def save_weights(self, path):
        """save the weights of the model

        Args:
            path (str): path to save the weights
        """
        torch.save(self.state_dict(), path)

    def _load_hparams(self, hparams):
        """loads the hparams from the file

        Args:
            hparams (str): the hparams path

        Returns:
            dict: the loaded hparams
        """
        return np.load(hparams, allow_pickle=True).tolist()

    def get_device(self):
        """get the device where the model is

        Returns:
            str: the device ('cpu' or 'cuda')
        """
        return next(self.parameters()).device


'''
    The model
'''

try:
    from .modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from .modelsTests.utils.Trainer_aic_transformer_lstm import Trainer_aic_transformer_lstm
except: # notebooks
    from stud.modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from stud.modelsTests.utils.Trainer_aic_transformer_lstm import Trainer_aic_transformer_lstm


import os

class ArgIdenClassModel():

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(
        self, 
        language: str,
        device = None,
        root_path = '../../../../',
        model_save_file_path = None,
        model_load_weights = True,
        loss_fn = None,
        fine_tune_transformer = False,
        saves_path_folder = 'test2',
    ):
        """the wrapper model for the argument identification + classification part

        Args:
            language (str): the language used for the test set.
            device (str, optional): the device in which the model needs to be loaded. Defaults to None.
            root_path (str, optional): root of the current environment. Defaults to '../../../../'.
            model_save_file_path (str, optional): the path to the weights of the model. Defaults to None.
            model_load_weights (bool, optional): if the model needs to load the weights. Defaults to True.
            loss_fn (any, optional): the loss function. Defaults to None.
            fine_tune_transformer (bool, optional): If the transformer needs to be fine-tuned. Defaults to False.
            saves_path_folder (str, optional): the path to the saves folder (for the parameters and other possible weights). Defaults to 'test2'.
        """

        self.trainer = Trainer_aic_transformer_lstm()

        # root:

        saves_path = os.path.join(root_path, f'model/{saves_path_folder}/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        # define principal paths:

        model_save_file_path = os.path.join(saves_path,'aic_transformer_lstm_fc_weights.pth') if model_save_file_path is None else model_save_file_path

        # load the specific model for the input language:

        self.language = language

        self.model = Net_aic_transformer_lstm( 
            hparams = self.hparams,
            loss_fn = loss_fn,
            fine_tune_transformer = fine_tune_transformer,
            load_transformer_config = model_load_weights,
        )

        if model_load_weights:
            self.model.load_weights(model_save_file_path)

        self.device = self.model.get_device() if device is None else device
        self.model.to(device)

        self.model.eval()
        
        # load vocabs:
        
        labels = SRLDataset_transformer.load_dict(os.path.join(saves_path, 'labels.npy'))
        self.dataset = SRLDataset_transformer(
            lang_data_path=None,
            tokenizer = self.hparams['transformer_name'],
            labels=labels,
        )
        self.dataset.collate_fn = self.dataset.create_collate_fn()

    def predict(self, sentence):
        predictions = {}
        predictions['roles'] = {}

        # for each predicate, split the sentence multiple times!
        sentences_preds = SRLDataset_transformer.split_sentence_by_predicates(sentence)

        if len(sentences_preds) == 0:
            return predictions

        # convert to ids
        sentence_preds_sample = self.dataset.collate_fn(sentences_preds)

        with torch.no_grad():
            dict_out = self.trainer.compute_forward(self.model, sentence_preds_sample, self.device, optimizer = None)
            sample_predictions = self.model.get_indices(dict_out['predictions']).detach().cpu().tolist()
            sample_predicate_position = dict_out['predicate_position']
            sample_output_mask = dict_out['output_mask']

            for i, (phrase_predicate_position, 
                    phrase_predictions, 
                    phrase_output_mask
                ) in enumerate(
                zip(sample_predicate_position, 
                    sample_predictions,
                    sample_output_mask)
            ):
                phrase_predictions_formatted = []
                for is_valid, prediction in zip(phrase_output_mask,phrase_predictions):
                    if is_valid:
                        phrase_predictions_formatted.append(prediction)

                predictions['roles'][phrase_predicate_position] = self.dataset.sentence_roles_converter(phrase_predictions_formatted, to='word')

        return predictions
