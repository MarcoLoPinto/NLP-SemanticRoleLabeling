import numpy as np

import json
from collections import Counter

from copy import deepcopy

try:
    from .SuperSRLDataset import SuperSRLDataset
except: # notebooks
    from stud.modelsTests.dataset.SuperSRLDataset import SuperSRLDataset

class SRLDataset_word_embedding(SuperSRLDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, 
        lang_data_path:str = None, 
        labels = None, 
        baselines_file_path = None, 
        split_predicates = True,
        word_vocabulary = None, 
        max_length_sentence = None):
        """

        Args:
            lang_data_path (str, optional): the path to a specific train or dev json dataset file. Defaults to None.
            labels (dict, optional): a dictionary containing the labels for each part of the project (if it's equal to None, then baselines_file_path is used). Defaults to None.
            baselines_file_path (str, optional): the path for the baselines file, used to generate the labels dictionary (if labels = None). Defaults to None.
            split_predicates (bool, optional): if True, each sentence is splitted in multiple sentences, one for each predicate in the sentence. Defaults to True.
            word_vocabulary (dict, optional): The vocabulary dictionary to convert word to id and vice-versa. Defaults to None.
            max_length_sentence (int, optional): Max length in a sentence. If None, it is computed from the dataset. Defaults to None.
        """
        
        super().__init__(lang_data_path, labels, baselines_file_path, split_predicates)
        
        self.word_vocabulary = word_vocabulary
        
        if lang_data_path is not None:
            self.data = SRLDataset_word_embedding.pad_data( 
                self.data, 
                pad_value=self.labels['roles_pad_token'] , 
                pad_len = max_length_sentence
            )
            self.pad_len = len( self.data[0]['words'] )

    @staticmethod
    def pad_data(data, pad_value = '<pad>', pad_len = None):
        """pads each sentence in the dataset

        Args:
            data (list): the dataset loaded wirth load_data()
            pad_value (str, optional): The pad string token. Defaults to '<pad>'.
            pad_len (int, optional): The length of a sentence. If None, it's the maximum length of a sentence in the dataset. Defaults to None.

        Returns:
            list: the padded dataset
        """
        data_padded = []
        max_sentence_words_len = len( max( data , key=lambda x:len(x['words']) )['words'] ) if pad_len is None else pad_len
        for i, sentence in enumerate(data):
            padded_sentence = deepcopy(sentence)
            padded_sentence['words'] = sentence['words'] + (max_sentence_words_len - len(sentence['words'])) * [pad_value]
            padded_sentence['lemmas'] = sentence['lemmas'] + (max_sentence_words_len - len(sentence['lemmas'])) * [pad_value]
            padded_sentence['predicates'] = sentence['predicates'] + (max_sentence_words_len - len(sentence['predicates'])) * [pad_value]
            padded_sentence['roles'] = sentence['roles'] + (max_sentence_words_len - len(sentence['roles'])) * [pad_value] # it is now a list
            data_padded.append( padded_sentence )
        return data_padded
    
    def sentence_words_converter(self, sentence_words, to = 'id'):
        """converts the words in the sentence from id to word and vice-versa

        Args:
            sentence_roles (list): the words in the sentece
            to (str, optional): if to='id' then it converts words to ids, otherwise the opposite. Defaults to 'id'.

        Raises:
            Exception: "to" parameter must be either 'id' or 'word'

        Returns:
            list: the encoded sentence
        """
        if not (to == 'id' or to == 'word'):
            raise Exception('Sorry, the parameter "to" must be either "id" or "word"!')
        vocab_type = 'key_to_index' if to == 'id' else 'index_to_key'
        unk_type = self.word_vocabulary['unk_id'] if to == 'id' else self.word_vocabulary['unk_token']
        
        encoded = [ self.word_vocabulary[vocab_type][word] 
                    if (
                        (to == 'id' and word in self.word_vocabulary[vocab_type]) or
                        (to == 'word' and word >= 0 and word < len(self.labels[vocab_type]))
                    ) 
                    else unk_type 
                    for word in sentence_words]
        return encoded

    def create_collate_fn(self):
        """ The collate_fn parameter for torch's DataLoader """
        def collate_fn(batch, to='id'):
            # These below are the inputs for the model
            words = [ self.sentence_words_converter(sentence['words'], to=to) for sentence in batch ] if (len(batch)>0 and 'words' in batch[0]) else []
            lemmas = [ self.sentence_words_converter(sentence['lemmas'], to=to) for sentence in batch ] if (len(batch)>0 and 'lemmas' in batch[0]) else []
            # These below are other inputs (maybe useful, but not inputted when doing predict)
            pos_tags = [ sentence['pos_tags'] for sentence in batch ] if (len(batch)>0 and 'pos_tags' in batch[0]) else []
            dependency_heads = [ sentence['dependency_heads'] for sentence in batch ] if (len(batch)>0 and 'dependency_heads' in batch[0]) else []
            dependency_relations = [ sentence['dependency_relations'] for sentence in batch ] if (len(batch)>0 and 'dependency_relations' in batch[0]) else []
            # This below can be the input for the model if we do argument identification + argument classification.
            # If we do also predicate disambiguation then the input will be only in 0s and 1s and this 
            # will be the desired output.
            # If we do also predicate identification then we don't have it and this will be the desired output.
            predicates = [ self.sentence_predicates_converter(sentence['predicates'], to=to) for sentence in batch ] if (len(batch)>0 and 'predicates' in batch[0]) else []
            predicate_position = [ sentence['predicate_position'] for sentence in batch ] if (len(batch)>0 and 'predicate_position' in batch[0]) else []
            # This below will be always in every case our desired output
            roles = [ self.sentence_roles_converter(sentence['roles'], to=to) for sentence in batch ] if (len(batch)>0 and 'roles' in batch[0]) else []

            # remember to properly convert them in tensors when passing them to the model
            return {'words':words, 'lemmas':lemmas, 'pos_tags':pos_tags, 
                    'dependency_heads':dependency_heads, 'dependency_relations':dependency_relations,
                    'predicates':predicates, 'predicate_position': predicate_position, 
                    'roles':roles}
            
        return collate_fn

    @staticmethod
    def get_predicates_distribution(data_raw):
        """generates the distribution of the data predicates

        Args:
            data_raw (list): the dataset

        Returns:
            Counter: a counter of the distributions
        """
        c = Counter()
        for sentence_details in data_raw:
            c.update( sentence_details['predicates'] )
        del c['_']
        return c

    @staticmethod
    def get_roles_distribution(data_raw):
        """generates the distribution of the data roles

        Args:
            data_raw (list): the dataset

        Returns:
            Counter: a counter of the distributions
        """
        c = Counter()
        for sentence_details in data_raw:
            if 'roles' in sentence_details: # some phrases has no roles!
                for sentence_predicate_roles in sentence_details['roles'].values():
                    c.update( sentence_predicate_roles )
        del c['_']
        return c
