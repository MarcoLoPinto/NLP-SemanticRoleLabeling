import torch
import numpy as np
import gensim.downloader as gensim_api

def generate_embedding_and_vocab(gensim_name = 'glove-wiki-gigaword-300', freeze = True, pad_token = '<pad>', unk_token = '<unk>'):
    """ Function that generates both word embedding and vocabulary from a gensim model. """
    embedding_keyedvectors = gensim_api.load(gensim_name)
    embedding_layer = create_embedding_from_keyedvectors(embedding_keyedvectors, freeze = freeze)
    vocabulary = create_vocabulary_from_keyedvectors(embedding_keyedvectors, pad_token = pad_token, unk_token = unk_token)
    return embedding_layer, vocabulary

def create_embedding_from_keyedvectors(keyedvectors, freeze = True):
    vectors = keyedvectors.vectors
    padding_idx = vectors.shape[0]
    # pad vector is all zeros
    pad = np.zeros((1, vectors.shape[1]))
    # unk vector is the mean vector
    unk = np.mean(vectors, axis=0, keepdims=True)
    # concatenate
    weights = torch.FloatTensor( np.concatenate((vectors,pad,unk)) )
    return torch.nn.Embedding.from_pretrained(weights, padding_idx=padding_idx, freeze=freeze)

def create_vocabulary_from_keyedvectors(keyedvectors, pad_token = '<pad>', unk_token = '<unk>'):
    index_to_key = keyedvectors.index_to_key.copy()
    key_to_index = keyedvectors.key_to_index.copy()
    
    index_to_key.append(pad_token)
    index_to_key.append(unk_token)

    key_to_index[pad_token] = len(key_to_index)
    key_to_index[unk_token] = len(key_to_index)

    return {
        'key_to_index':key_to_index, 'index_to_key':index_to_key, 
        'pad_token':pad_token, 'unk_token':unk_token, 
        'pad_id':key_to_index[pad_token], 'unk_id':key_to_index[unk_token],
    }

def save_embedding_layer(embedding, path):
    torch.save(embedding.state_dict(), path)
