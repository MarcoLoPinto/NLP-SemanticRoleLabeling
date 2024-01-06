'''
    The Net part
'''

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoConfig

class Net_pid_transformer_simple(nn.Module):
    """ predicate identification + predicate disamb model """
    def __init__(   self, 
                    fine_tune_transformer = False, hparams = {}, 
                    loss_fn = None, load_transformer_config = False, 
                    has_predicates_positions = True, has_pos_tags = False):
        """predicate identification + predicate disambiguation model

        Args:
            fine_tune_transformer (bool, optional): if the transformer weights needs to be fine-tuned or freezed. Defaults to False.
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
            load_transformer_config (bool, optional): if the model needs to be entirely loaded or just the configuration (used to speed-up the loading process in the evaluation part). Defaults to False.
            has_predicates_positions (bool, optional): if the model recevies the predicates for each word. Defaults to True.
            has_pos_tags (bool, optional): if the model recevies the part-of-speech for each word. Defaults to False.
        """
                    
        super().__init__()

        hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.n_labels = int(hparams['n_predicates_labels'])

        self.has_predicates_positions = has_predicates_positions
        self.has_pos_tags = has_pos_tags
        
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

        # 2) Classifier

        self.classifier = nn.Linear(transformer_out_dim + 1*has_pos_tags*self.n_labels, self.n_labels)

        # Loss function:
        self.loss_fn = loss_fn
    
    def forward(
        self, 
        input_ids, 
        attention_mask,  
        batch_predicates_positions = None,
        batch_pos_tags = None,
        token_type_ids = None,
    ):
        """_summary_

        Args:
            input_ids (torch.Tensor): the tensors inputs generated via the Tokenizer
            attention_mask (torch.Tensor): attention mask generated via the Tokenizer
            batch_predicates_positions (torch.Tensor, optional): the positions in the sentence of the predicates (0s and 1s). Has shape (batch). Defaults to None.
            batch_pos_tags (torch.Tensor, optional): the pos tag for each word. Defaults to None.
            token_type_ids (any, optional): generated via the Tokenizer. Defaults to None.

        Returns:
            _type_: _description_
        """

        delta_factor = 0 # summing factor for batch_predicates_positions

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

        if self.has_pos_tags:
            # For each POS in the sentence it associates a one-hot representation
            # for that particular word.
            # If the value is equal to -1, then the encoding for it becomes [0,0,...,0]
            # (i.e. that particular word in the sentence is padding, so convert to a vector of zeros)
            batch_sentence_words = torch.cat((
                batch_sentence_words,
                torch.nn.functional.one_hot( 
                    torch.where(batch_pos_tags >=0 , batch_pos_tags , 0), 
                    num_classes = self.n_labels,
                ) * (batch_pos_tags >= 0)[:,:,None]
            ), dim=-1)

        # if self.has_predicates_positions:
        #     batch_sentence_words = torch.cat((
        #         batch_sentence_words, 
        #         batch_predicates_positions.unsqueeze(-1),
        #     ), dim=-1)

        # 2) Classifier

        if self.has_predicates_positions:
            batch_sentence_words = batch_sentence_words * (batch_predicates_positions + delta_factor).unsqueeze(-1)
        
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
            torch_outputs (torch.Tensor): a Tensor with shape (batch_size, sentence_len, label_vocab_size) containing the logits outputed by the neural network (if batch_first = True)
        
        Returns:
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

# TODO!!! model!

'''
    The model
'''

try:
    from .modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from .modelsTests.utils.Trainer_pid_transformer_simple import Trainer_pid_transformer_simple
except: # notebooks
    from stud.modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from stud.modelsTests.utils.Trainer_pid_transformer_simple import Trainer_pid_transformer_simple


import os

class PredIdenDisModel():

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
        has_predicates_positions = False,
        has_pos_tags = False,
        saves_path_folder = 'test4',
        tokenizer = None,
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
            has_predicates_positions (bool, optional): if the model recevies the predicates for each word. Defaults to True.
            has_pos_tags (bool, optional): if the model recevies the part-of-speech for each word. Defaults to False.
            saves_path_folder (str, optional): the path to the saves folder (for the parameters and other possible weights). Defaults to 'test3'.
            tokenizer (any, optional): the tokenizer to use for the model. Defaults to None.
        """

        self.trainer = Trainer_pid_transformer_simple()

        # root:

        saves_path = os.path.join(root_path, f'model/{saves_path_folder}/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        # define principal paths:

        model_name = f'pd_transformer_weights_{language.lower()}.pth' if has_predicates_positions else f'pid_transformer_weights_{language.lower()}.pth'

        model_save_file_path = os.path.join(saves_path,model_name) if model_save_file_path is None else model_save_file_path

        # load the specific model for the input language:

        self.language = language

        self.model = Net_pid_transformer_simple( 
            hparams = self.hparams,
            loss_fn = loss_fn,
            fine_tune_transformer = fine_tune_transformer,
            load_transformer_config = model_load_weights,
            has_predicates_positions = has_predicates_positions,
            has_pos_tags = has_pos_tags,
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
            tokenizer = self.hparams['transformer_name'] if tokenizer is None else tokenizer,
            labels=labels,
            split_predicates = False, # !
        )
        self.dataset.collate_fn = self.dataset.create_collate_fn()

    def predict(self, sentence):
        predictions = {'predicates':[]}

        # convert to ids
        sentence_preds_sample = self.dataset.collate_fn([sentence])

        with torch.no_grad():
            dict_out = self.trainer.compute_forward(self.model, sentence_preds_sample, self.device, optimizer = None)
            sample_predictions = self.model.get_indices(dict_out['predictions']).detach().cpu().tolist()[0]
            sample_output_mask = dict_out['output_mask'][0]

            phrase_predictions_formatted = []
            for is_valid, prediction in zip(sample_output_mask,sample_predictions):
                if is_valid:
                    phrase_predictions_formatted.append(prediction)

            predictions['predicates'] = self.dataset.sentence_predicates_converter(phrase_predictions_formatted, to='word')

        return predictions
