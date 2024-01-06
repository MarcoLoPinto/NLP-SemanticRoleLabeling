import numpy as np

import json
from collections import Counter

from copy import deepcopy

try:
    from .SuperSRLDataset import SuperSRLDataset
except: # notebooks
    from stud.modelsTests.dataset.SuperSRLDataset import SuperSRLDataset

from transformers import AutoTokenizer
import torch

class SRLDataset_transformer(SuperSRLDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, 
        lang_data_path:str = None, 
        labels = None, 
        baselines_file_path = None, 
        split_predicates = True,
        tokenizer = None):
        """

        Args:
            lang_data_path (str, optional): the path to a specific train or dev json dataset file. Defaults to None.
            labels (dict, optional): a dictionary containing the labels for each part of the project (if it's equal to None, then baselines_file_path is used). Defaults to None.
            baselines_file_path (str, optional): the path for the baselines file, used to generate the labels dictionary (if labels = None). Defaults to None.
            split_predicates (bool, optional): if True, each sentence is splitted in multiple sentences, one for each predicate in the sentence. Defaults to True.
            tokenizer (any, optional): The tokenizer (or the name of the tokenizer) used for this dataset. Defaults to None.
        """
        super().__init__(lang_data_path, labels, baselines_file_path, split_predicates)

        self.split_predicates = split_predicates
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer

    def create_collate_fn(self):
        """ The collate_fn parameter for torch's DataLoader """
        if self.split_predicates: # this is for argument iden-class
            def collate_fn(batch):

                # tokenizing the words - predicate pairs
                batch_words_pair = [ sentence['words'] for sentence in batch ]
                batch_predicates_pair = [ [sentence['predicate_word']] for sentence in batch ]
                batch_formatted = self.tokenizer(
                    batch_words_pair, batch_predicates_pair,
                    return_tensors="pt",
                    padding=True,
                    is_split_into_words=True, 
                )
                
                # converting each list of roles in the sentence in integers
                batch_roles = [self.sentence_roles_converter(sentence['roles'], to='id') for sentence in batch] if (len(batch)>0 and 'roles' in batch[0]) else [] # roles labels, for the target predicate
                batch_predicates = [ self.sentence_predicates_converter(sentence['predicates'], to='id') for sentence in batch ]
                batch_pos_tags = [ self.sentence_pos_converter(sentence['pos_tags'], to='id') for sentence in batch ] if (len(batch)>0 and 'pos_tags' in batch[0]) else []

                batch_predicate_position = [ sentence['predicate_position'] for sentence in batch ]
                batch_predicate_position_formatted = deepcopy(batch_predicate_position)

                batch_roles_formatted = []
                batch_predicates_formatted = []
                batch_pos_tags_formatted = []

                matrix_subwords = torch.as_tensor(
                    len(batch_formatted['input_ids'])*[len(batch_formatted['input_ids'][0])*[len(batch_formatted['input_ids'][0])*[0]]]
                )

                output_mask = []

                for i, sample in enumerate(batch):
                    words_ids = batch_formatted.word_ids(batch_index=i)
                    
                    previous_word_id = None
                    nones_seen = 0

                    roles_formatted = []
                    predicates_formatted = []
                    pos_tags_formatted = []

                    matrix_subwords_sentence_last_index = 0

                    output_mask_row = []

                    for j, word_id in enumerate(words_ids):

                        if word_id is None: # Special tokens have the id = None. Setting the label to -1 so they are ignored by the loss
                            if batch_roles != []:
                                roles_formatted.append( self.labels['roles_pad_id'] )

                            predicates_formatted.append( self.labels['predicates_pad_id'] )
                            if batch_pos_tags != []:
                                pos_tags_formatted.append( self.labels['pos_pad_id'] )
                            
                            batch_predicate_position_formatted[i] += 1 if nones_seen <= 1 and j <= batch_predicate_position_formatted[i] else 0

                            matrix_subwords_sentence_last_index = 0

                            output_mask_row.append(0)

                            nones_seen += 1

                        elif word_id != previous_word_id: # if differs, it's not a subword of the previous word
                            if batch_roles != []:
                                if word_id < len(batch_roles[i]) and nones_seen <= 1:
                                    roles_formatted.append( batch_roles[i][word_id] )
                                else:
                                    roles_formatted.append( self.labels['roles_pad_id'] )

                            if nones_seen <= 1:
                                predicates_formatted.append( batch_predicates[i][word_id] )
                                if batch_pos_tags != []:
                                    pos_tags_formatted.append( batch_pos_tags[i][word_id] )
                            else:
                                predicates_formatted.append( self.labels['predicates_pad_id'] )
                                if batch_pos_tags != []:
                                    pos_tags_formatted.append( self.labels['pos_pad_id'] )

                            matrix_subwords_sentence_last_index = 0
                            matrix_subwords[i][j][j] = 1 if nones_seen <= 1 else 0

                            output_mask_row.append(1 if nones_seen <= 1 else 0)

                        else: # if it's a subword!
                            if batch_roles != []:
                                roles_formatted.append( self.labels['roles_pad_id'] ) # subword ignored!

                            predicates_formatted.append( self.labels['predicates_pad_id'] )
                            if batch_pos_tags != []:
                                pos_tags_formatted.append( self.labels['pos_pad_id'] )

                            batch_predicate_position_formatted[i] += 1 if nones_seen <= 1 and j <= batch_predicate_position_formatted[i] else 0

                            matrix_subwords_sentence_last_index += 1
                            matrix_subwords[i][j - matrix_subwords_sentence_last_index][j] = 1 if nones_seen <= 1 else 0

                            output_mask_row.append(0)

                        previous_word_id = word_id
                    
                    # pad rest of the sentence:
                    # sentence_roles_formatted += [ self.labels['roles_pad_id'] ] * ( len(batch_formatted['input_ids'][i]) - len(sentence_roles_formatted) )
                    batch_roles_formatted.append(roles_formatted)
                    batch_predicates_formatted.append(predicates_formatted)
                    batch_pos_tags_formatted.append(pos_tags_formatted)

                    # output_mask_row += [0] * ( len(batch_formatted['input_ids'][i]) - len(output_mask_row) )
                    output_mask.append(output_mask_row)

                batch_formatted['roles_formatted'] = torch.as_tensor(batch_roles_formatted)

                batch_formatted['matrix_subwords'] = matrix_subwords

                batch_formatted['output_mask'] = torch.as_tensor(output_mask)

                batch_formatted['predicates'] = torch.as_tensor(batch_predicates_formatted)
                batch_formatted['predicate_position'] = batch_predicate_position
                batch_formatted['predicate_position_formatted'] = torch.as_tensor(batch_predicate_position_formatted)

                batch_formatted['pos_tags'] = torch.as_tensor(batch_pos_tags_formatted)

                return batch_formatted
            
            return collate_fn

        else: # this is for predicate iden-diamb
            def collate_fn(batch):

                # tokenizing the words
                batch_words_pair = [ sentence['words'] for sentence in batch ]
                batch_formatted = self.tokenizer(
                    batch_words_pair,
                    return_tensors="pt",
                    padding=True,
                    is_split_into_words=True, 
                )
                
                if (len(batch)>0 and 'predicates' in batch[0]):
                    if type(batch[0]['predicates'][0]) == int:
                        batch_predicates = [ sentence['predicates'] for sentence in batch ]
                    else:
                        batch_predicates = [ self.sentence_predicates_converter(sentence['predicates'], to='id') for sentence in batch ]
                else:
                    batch_predicates = []
                batch_pos_tags = [ self.sentence_pos_converter(sentence['pos_tags'], to='id') for sentence in batch ] if (len(batch)>0 and 'pos_tags' in batch[0]) else []

                batch_predicates_formatted = []
                batch_pos_tags_formatted = []

                matrix_subwords = torch.as_tensor(
                    len(batch_formatted['input_ids'])*[len(batch_formatted['input_ids'][0])*[len(batch_formatted['input_ids'][0])*[0]]]
                )

                output_mask = []

                for i, sample in enumerate(batch):
                    words_ids = batch_formatted.word_ids(batch_index=i)
                    
                    previous_word_id = None
                    nones_seen = 0

                    predicates_formatted = []
                    pos_tags_formatted = []

                    matrix_subwords_sentence_last_index = 0

                    output_mask_row = []

                    for j, word_id in enumerate(words_ids):

                        if word_id is None: # Special tokens have the id = None. Setting the label to -1 so they are ignored by the loss

                            predicates_formatted.append( self.labels['predicates_pad_id'] )
                            if batch_pos_tags != []:
                                pos_tags_formatted.append( self.labels['pos_pad_id'] )

                            matrix_subwords_sentence_last_index = 0

                            output_mask_row.append(0)

                            nones_seen += 1

                        elif word_id != previous_word_id: # if differs, it's not a subword of the previous word

                            if nones_seen <= 1:
                                if batch_predicates != []:
                                    predicates_formatted.append( batch_predicates[i][word_id] )
                                if batch_pos_tags != []:
                                    pos_tags_formatted.append( batch_pos_tags[i][word_id] )
                            else:
                                predicates_formatted.append( self.labels['predicates_pad_id'] )
                                pos_tags_formatted.append( self.labels['pos_pad_id'] )

                            matrix_subwords_sentence_last_index = 0
                            matrix_subwords[i][j][j] = 1 if nones_seen <= 1 else 0

                            output_mask_row.append(1 if nones_seen <= 1 else 0)

                        else: # if it's a subword!

                            predicates_formatted.append( self.labels['predicates_pad_id'] )
                            if batch_pos_tags != []:
                                pos_tags_formatted.append( self.labels['pos_pad_id'] )

                            matrix_subwords_sentence_last_index += 1
                            matrix_subwords[i][j - matrix_subwords_sentence_last_index][j] = 1 if nones_seen <= 1 else 0

                            output_mask_row.append(0)

                        previous_word_id = word_id
                    
                    # pad rest of the sentence:
                    batch_predicates_formatted.append(predicates_formatted)
                    batch_pos_tags_formatted.append(pos_tags_formatted)

                    # output_mask_row += [0] * ( len(batch_formatted['input_ids'][i]) - len(output_mask_row) )
                    output_mask.append(output_mask_row)

                batch_formatted['matrix_subwords'] = matrix_subwords

                batch_formatted['output_mask'] = torch.as_tensor(output_mask)

                batch_formatted['predicates'] = torch.as_tensor(batch_predicates_formatted)

                batch_formatted['pos_tags'] = torch.as_tensor(batch_pos_tags_formatted)

                batch_formatted['predicates_binary'] = torch.where(
                    torch.as_tensor(batch_predicates_formatted) > 0, 1, 0
                )

                return batch_formatted
            
            return collate_fn
