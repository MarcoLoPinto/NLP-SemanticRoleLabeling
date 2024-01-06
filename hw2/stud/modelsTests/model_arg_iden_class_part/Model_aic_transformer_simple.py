'''
    The Net part
'''

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoConfig

class Net_aic_transformer_simple(nn.Module):
    """ argument identification + argument classification model """
    def __init__(self, fine_tune_transformer = False, hparams = {}, loss_fn = None, load_transformer_config = False, n_transformer_hidden_states = 4):
        """argument identification + argument classification model

        Args:
            fine_tune_transformer (bool, optional): if the transformer weights needs to be fine-tuned or freezed. Defaults to False.
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
            load_transformer_config (bool, optional): if the model needs to be entirely loaded or just the configuration (used to speed-up the loading process in the evaluation part). Defaults to False.
            n_transformer_hidden_states (int, optional): how many of the last layers of the transformer needs to be summed and passed on the classification layer. Defaults to 4.
        """
        super().__init__()

        hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.n_transformer_hidden_states = n_transformer_hidden_states

        self.n_labels = int(hparams['n_roles_labels'])
        # self.n_predicates = int(hparams['n_predicates_labels'])
        
        # layers:

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
                param.requires_grad = False

        transformer_out_dim = self.transformer_model.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(transformer_out_dim, self.n_labels)

        # Loss function:
        self.loss_fn = loss_fn
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids = None,
    ):
        """forward function

        Args:
            input_ids (torch.Tensor): the tensors inputs generated via the Tokenizer
            attention_mask (torch.Tensor): attention mask generated via the Tokenizer
            token_type_ids (any, optional): generated via the Tokenizer. Defaults to None.

        Returns:
            torch.Tensor: the logits
        """

        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if token_type_ids is None: # some transformer models don't have it
            transformer_kwargs['token_type_ids'] = token_type_ids

        n_transformer_hidden_states = self.n_transformer_hidden_states

        transformer_outs = self.transformer_model(**transformer_kwargs)

        transformer_out = torch.stack(
            transformer_outs.hidden_states[-n_transformer_hidden_states:],
            dim=0).sum(dim=0)
        
        logits = self.classifier(transformer_out)

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


'''
    The model
'''

try:
    from .modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from .modelsTests.utils.Trainer_aic_transformer_simple import Trainer_aic_transformer_simple
except: # notebooks
    from stud.modelsTests.dataset.SRLDataset_transformer import SRLDataset_transformer
    from stud.modelsTests.utils.Trainer_aic_transformer_simple import Trainer_aic_transformer_simple


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
        saves_path_folder = 'test4',
        n_transformer_hidden_states = 4,
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
            saves_path_folder (str, optional): the path to the saves folder (for the parameters and other possible weights). Defaults to 'test2'.
            tokenizer (any, optional): the tokenizer to use for the model. Defaults to None.
        """

        self.trainer = Trainer_aic_transformer_simple()

        # root:

        saves_path = os.path.join(root_path, f'model/{saves_path_folder}/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        # define principal paths:

        model_name = f'aic_transformer_weights_{language.lower()}.pth'

        model_save_file_path = os.path.join(saves_path,model_name) if model_save_file_path is None else model_save_file_path

        # load the specific model for the input language:

        self.language = language

        self.model = Net_aic_transformer_simple( 
            hparams = self.hparams,
            loss_fn = loss_fn,
            fine_tune_transformer = fine_tune_transformer,
            load_transformer_config = model_load_weights,
            n_transformer_hidden_states = n_transformer_hidden_states,
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
