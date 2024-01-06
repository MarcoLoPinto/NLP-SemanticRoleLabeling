'''
    The Net part
'''

import torch
import torch.nn as nn
import numpy as np

class Net_aic_lstm(nn.Module):
    """ argument identification + argument classification model """
    def __init__(self, word_embedding = None, hparams = {}, loss_fn = None):
        """argument identification + argument classification model

        Args:
            word_embedding (any, optional): Word Embedding save path or the generated Embedding. Defaults to None.
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
        """
        super().__init__()

        hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.n_predicates = int(hparams['n_predicates_labels'])
        self.n_labels = int(hparams['n_roles_labels'])
        predicates_mask_dim = 1

        # 1) Embeddings
        if word_embedding is not None:
            self.word_embedding = word_embedding if type(word_embedding) != str else self._load_embedding(word_embedding, hparams)
        else:
            self.word_embedding = torch.nn.Embedding(
                hparams['embedding_word_shape'][0],
                hparams['embedding_word_shape'][1], 
                padding_idx=hparams['embedding_word_padding_idx'])

        self.word_embedding.weight.requires_grad_(hparams['embedding_word_requires_grad'])

        word_embedding_dim_out = self.word_embedding.embedding_dim

        # 2) Sequence encoder

        word_lstm_n_layers = 3
        word_lstm_hidden_size = 256
        word_lstm_bidirectional = True
        self.word_lstm = nn.LSTM(
            input_size = word_embedding_dim_out + self.n_predicates + predicates_mask_dim, 
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

        self.classifier = nn.Linear(word_lstm_dim_out // 2, self.n_labels)

        # Loss function:
        self.loss_fn = loss_fn

        # Extras
        self.relu = nn.ReLU()
    
    def forward(self, batch_sentence_words, batch_sentence_predicates, batch_predicate_position):
        """forward function

        Args:
            batch_sentence_words (torch.Tensor): the words in the sentence. It has shape (batch, sentence_len)
            batch_sentence_predicates (torch.Tensor): it has the predicates labels for each word (in the training/test, there is only one for sentence!). It has shape (batch, sentence_len)
            batch_predicate_position (torch.Tensor): the position in the sentence of the target predicate. It has shape (batch)

        Returns:
            torch.Tensor: the logits
        """

        # For each sentence in the batch, it generates an array of the same length as the sentence:
        # the only 1 in each array corresponds to the position of the predicate in the sentence.
        # If the position of the target predicate is equal to -1, then the encoding for it becomes [0,0,...,0] 
        # (i.e. there is no target predicate for that sentence).
        # predicate_mask has shape (batch, sentence_len)
        predicate_mask = torch.nn.functional.one_hot( 
            torch.where(batch_predicate_position >=0 , batch_predicate_position , 0) , 
            num_classes=batch_sentence_words.shape[-1] 
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

        batch_sentence_words = self.word_embedding(batch_sentence_words) # -> (batch, sentence_len, word_emb_dim)
        
        batch_sentence_words = torch.cat((
            batch_sentence_words, 
            batch_sentence_predicates_onehot, 
            predicate_mask.unsqueeze(-1),
        ), dim=-1)

        batch_sentence_words, _ = self.word_lstm(batch_sentence_words)

        batch_sentence_words = self.fc1(batch_sentence_words)
        batch_sentence_words = self.ln1(batch_sentence_words)
        batch_sentence_words = self.relu(batch_sentence_words)

        batch_sentence_words = self.classifier(batch_sentence_words) 

        return batch_sentence_words # (batch, sentence_len, n_lables)

    def compute_loss(self, x, y_true):
        """computes the loss for the net

        Args:
            x (torch.Tensor): The predictions
            y_true (torch.Tensor): The true labels

        Returns:
            any: the loss
        """
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

    def _load_embedding(self, path, hparams):
        """load the embedding

        Args:
            path (str): path to the saved embedding
            hparams (dict): parameters necessary to initialize the embedding

        Returns:
            The torch.nn.Embedding loaded layer
        """
        word_embedding = torch.nn.Embedding(hparams['embedding_word_shape'][0],hparams['embedding_word_shape'][1], padding_idx=hparams['embedding_word_padding_idx'])
        weights = torch.load(path)
        word_embedding.load_state_dict(weights)
        return word_embedding

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
    from .modelsTests.dataset.SRLDataset_word_embedding import SRLDataset_word_embedding
    from .modelsTests.utils.Trainer_aic_lstm import Trainer_aic_lstm
except: # notebooks
    from stud.modelsTests.dataset.SRLDataset_word_embedding import SRLDataset_word_embedding
    from stud.modelsTests.utils.Trainer_aic_lstm import Trainer_aic_lstm


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
        saves_path_folder = 'test1',
    ):
        """the wrapper model for the argument identification + classification part

        Args:
            language (str): the language used for the test set.
            device (str, optional): the device in which the model needs to be loaded. Defaults to None.
            root_path (str, optional): root of the current environment. Defaults to '../../../../'.
            model_save_file_path (str, optional): the path to the weights of the model. Defaults to None.
            model_load_weights (bool, optional): if the model needs to load the weights. Defaults to True.
            loss_fn (any, optional): the loss function. Defaults to None.
            saves_path_folder (str, optional): the path to the saves folder (for the parameters and other possible weights). Defaults to 'test1'.
        """

        self.trainer = Trainer_aic_lstm()

        # root:

        saves_path = os.path.join(root_path, f'model/{saves_path_folder}/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        # define principal paths:

        model_save_file_path = os.path.join(saves_path,'arg_iden_class_net_weights-81-75.pth') if model_save_file_path is None else model_save_file_path

        # load the specific model for the input language:

        self.language = language

        self.model = Net_aic_lstm( 
            hparams = self.hparams,
            loss_fn = loss_fn,
        )
        if model_load_weights:
            self.model.load_weights(model_save_file_path)

        self.device = self.model.get_device() if device is None else device

        if loss_fn is None:
            self.model.to(device)
            self.model.eval()
        
        # load vocabs:

        word_vocabulary = SRLDataset_word_embedding.load_dict(os.path.join(saves_path, 'EN_word_vocabulary.npy'))
        labels = SRLDataset_word_embedding.load_dict(os.path.join(saves_path, 'labels.npy'))
        self.dataset = SRLDataset_word_embedding(
            lang_data_path=None,
            word_vocabulary=word_vocabulary,
            labels=labels,
            max_length_sentence=self.hparams['sentence_len']
        )
        self.dataset.collate_fn = self.dataset.create_collate_fn()

    def predict(self, sentence):
        predictions = {}
        predictions['roles'] = {}

        # for each predicate, split the sentence multiple times!
        sentences_preds = SRLDataset_word_embedding.split_sentence_by_predicates(sentence)

        if len(sentences_preds) == 0:
            return predictions

        # convert to ids
        sentence_preds_sample = self.dataset.collate_fn(sentences_preds, to='id')

        with torch.no_grad():
            sample_out = self.trainer.compute_forward(self.model, sentence_preds_sample, self.device, optimizer = None)
            sample_predictions = self.model.get_indices(sample_out['predictions'])
            sample_predictions = sample_predictions.detach().cpu().tolist()
            sample_predicate_position = sentence_preds_sample['predicate_position']

            for i, (phrase_predictions, phrase_predicate_position) in enumerate(zip(sample_predictions, sample_predicate_position)):
                predictions['roles'][phrase_predicate_position] = self.dataset.sentence_roles_converter(phrase_predictions, to = 'word') 

        return predictions