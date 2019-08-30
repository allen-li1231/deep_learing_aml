import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import re
import random
from random import choice

choice.
IMG_SIZE = 299

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

train_captions=[['A long dirt road going through a forest.',
                 'A SCENE OF WATER AND A PATH WAY',
                 'A sandy path surrounded by trees leads to a beach.',
                 'Ocean view through a dirt road surrounded by a forested area. ',
                 'dirt path leading beneath barren trees to open plains'],
                ['A group of zebra standing next to each other.',
                 'This is an image of of zebras drinking',
                 'ZEBRAS AND BIRDS SHARING THE SAME WATERING HOLE',
                 'Zebras that are bent over and drinking water together.',
                 'a number of zebras drinking water near one another']]

# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more,
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary),
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    vocab = set()
    d={}
    for caption in train_captions:

        for tokens in caption:
            for t in split_sentence(tokens):
                if d.get(t) is None:
                    d[t] = 1
                else:
                    d[t] += 1

        vocab = vocab | set(word for word in d.keys() if d[word] >= 5)
    vocab = vocab | set([PAD, UNK, START, END])
    return {token: index for index, token in enumerate(sorted(vocab))}


def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    res = []
    for c in captions:
        outer_container = []
        for sentence in c:
            container = []
            container.append(vocab[START])
            for t in split_sentence(sentence):
                if t in vocab.keys():
                    container.append(vocab[t])
                else:
                    container.append(vocab[UNK])
            container.append(vocab[END])
            outer_container.append(container)
        res.append(outer_container)
    return res


# prepare vocabulary
vocab = generate_vocabulary(train_captions)
vocab_inverse = {idx: w for w, idx in vocab.items()}
print(len(vocab))

# replace tokens with indices
train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)


# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    max_in_caption = max(map(len, batch_captions))
    container = list(map(lambda x: x + [pad_idx,] * (max_in_caption-len(x)), batch_captions))
    matrix = np.array(container)###YOUR CODE HERE###
    if max_len and max_len < max_in_caption:
        matrix = matrix[:, :max_len]
    return matrix


IMG_EMBED_SIZE = train_img_embeds.shape[1]
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]