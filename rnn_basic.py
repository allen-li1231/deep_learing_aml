import numpy as np
import tensorflow as tf
import keras
from keras.layers import concatenate, Dense, Embedding
import matplotlib.pyplot as plt
from random import sample


start_token = " "  # so that the network knows that we're generating a first token

# this is the token for padding,
# we will add fake pad token at the end of names
# to make them of equal size for further batching
pad_token = "!"

with open("names.txt") as f:
    names = f.read()[:-1].split('\n')
    names = [start_token + name for name in names]


MAX_LENGTH = max(map(len, names))
print("max length:", MAX_LENGTH)

plt.title('Sequence length distribution')
plt.hist(list(map(len, names)), bins=25)

# generate a dictionary that encode inputs as a sequence of character ids
tokens = [chr(i) for i in range(ord('a'), ord('z')+1)]
tokens = tokens + [chr(i) for i in range(ord('A'), ord('Z')+1)] + [' ', '-', '\'', '!']
n_tokens = len(tokens)
token_to_id = {symbol: idx for idx, symbol in enumerate(tokens)}


def to_matrix(names, max_len=None, pad=token_to_id[pad_token], dtype=np.int32):
    """Casts a list of names into rnn-digestable padded matrix"""

    max_len = max_len or max(map(len, names))
    names_ix = np.zeros([len(names), max_len], dtype) + pad
    for i in range(len(names)):
        name_ix = list(map(token_to_id.get, names[i]))
        names_ix[i, :len(name_ix)] = name_ix

    return names_ix


rnn_num_units = 64  # size of hidden state
embedding_size = 16  # for characters

# Let's create layers for our recurrent network
# Note: we create layers but we don't "apply" them yet (this is a "functional API" of Keras)
# Note: set the correct activation (from keras.activations) to Dense layers!

# an embedding layer that converts character ids into embeddings
# The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:

# It must specify 3 arguments:

# input_dim: This is the size of the vocabulary in the text data.
# For example, if your data is integer encoded to values between 0-10,
# then the size of the vocabulary would be 11 words.
# output_dim: This is the size of the vector space in which words will be embedded.
# It defines the size of the output vectors from this layer for each word.
# For example, it could be 32 or 100 or even larger. Test different values for your problem.
# input_length: This is the length of input sequences as you would define for any input layer of a Keras model.
# For example, if all of your input documents are comprised of 1000 words, this would be 1000.
embed_x = Embedding(n_tokens, embedding_size)

# a dense layer that maps input and previous state to new hidden state, [x_t,h_t]->h_t+1
get_h_next = Dense(rnn_num_units, activation='relu')

# a dense layer that maps current hidden state to probabilities of characters [h_t+1]->P(x_t+1|h_t+1)
get_probas = Dense(n_tokens, activation='sigmoid')


def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces
    probabilities for next token x_t+1 and next state h_t+1
    given current input x_t and previous state h_t.
    We'll call this method repeatedly to produce the whole sequence.

    You're supposed to "apply" above layers to produce new tensors.
    Follow inline instructions to complete the function.
    """
    # convert character id into embedding
    x_t_emb = embed_x(tf.reshape(x_t, [-1, 1]))[:, 0]

    # concatenate x_t embedding and previous h_t state
    x_and_h = concatenate([x_t_emb, h_t])

    # compute next state given x_and_h
    h_next = get_h_next(x_and_h)

    # get probabilities for language model P(x_next|h_next)
    output_probas = get_probas(h_next)

    return output_probas, h_next


input_sequence = tf.placeholder(tf.int32, (None, MAX_LENGTH))  # batch of token ids
batch_size = tf.shape(input_sequence)[0]

predicted_probas = []
h_prev = tf.zeros([batch_size, rnn_num_units])  # initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[:, t]  # column t
    probas_next, h_next = rnn_one_step(x_t, h_prev)

    h_prev = h_next
    predicted_probas.append(probas_next)

# combine predicted_probas into [batch, time, n_tokens] tensor
predicted_probas = tf.transpose(tf.stack(predicted_probas), [1, 0, 2])

# next to last token prediction is not needed
predicted_probas = predicted_probas[:, :-1, :]

# flatten predictions to [batch*time, n_tokens]
predictions_matrix = tf.reshape(predicted_probas, [-1, n_tokens])

# flatten answers (next tokens) and one-hot encode them
answers_matrix = tf.one_hot(tf.reshape(input_sequence[:, 1:], [-1]), n_tokens)

# Define the loss as categorical cross-entropy (e.g. from keras.losses).
loss = tf.reduce_mean(keras.losses.categorical_crossentropy(answers_matrix, predictions_matrix))

optimize = tf.train.AdamOptimizer().minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

s = tf.Session(config=config)
s.run(tf.global_variables_initializer())

batch_size = 32
history = []

for i in range(1000):
    batch = to_matrix(sample(names, batch_size), max_len=MAX_LENGTH)
    loss_i, _ = s.run([loss, optimize], {input_sequence: batch})

    history.append(loss_i)

    if (i + 1) % 100 == 0:
        plt.plot(history, label='loss')
        plt.legend()
        plt.show()


x_t = tf.placeholder(tf.int32, (1,))
h_t = tf.Variable(np.zeros([1, rnn_num_units], np.float32))  # we will update hidden state in this variable

# For sampling we need to define `rnn_one_step` tensors only once in our graph.
# We reuse all parameters thanks to functional API usage.
# Then we can feed appropriate tensor values using feed_dict in a loop.
# Note how different it is from training stage, where we had to unroll the whole sequence for backprop.
next_probs, next_h = rnn_one_step(x_t, h_t)


def generate_sample(seed_phrase=start_token, max_length=MAX_LENGTH):
    """
    This function generates text given a `seed_phrase` as a seed.
    Remember to include start_token in seed phrase!
    Parameter `max_length` is used to set the number of characters in prediction.
    """
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t, h_t.initial_value))

    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        s.run(tf.assign(h_t, next_h), {x_t: [ix]})

    # start generating
    for _ in range(max_length-len(seed_phrase)):
        x_probs, _ = s.run([next_probs, tf.assign(h_t, next_h)], {x_t: [x_sequence[-1]]})
        x_probs = x_probs / x_probs.sum()
        x_sequence.append(np.random.choice(n_tokens, p=x_probs[0]))

    return ''.join([tokens[ix] for ix in x_sequence if tokens[ix] != pad_token])


# show off the training result of the model without prefix
for _ in range(10):
    print(generate_sample())

# with prefix conditioning
for _ in range(10):
    print(generate_sample(' Trump'))
