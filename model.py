import numpy as np
np.random.seed(42)

from keras.layers import Conv2D, MaxPooling2D, Embedding, GRU
from keras.layers import TimeDistributed
from keras.layers import Dense, Input, Flatten, Lambda, Softmax, Dot
from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers
import tensorflow as tf
tf.set_random_seed(42)

class StaticWeights(Layer):
    """ Utterance-Response Matching By Words Layer """
    def __init__(self,
                 last_dim=50,
                 max_turn=10,
                 **kwargs):
        self.last_dim = last_dim
        self.max_turn = max_turn
        super(StaticWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight((self.max_turn, 1), name='w', initializer=initializers.Ones(), trainable=True)
        super(StaticWeights, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(tf.multiply(inputs, self.w), axis=1)  # (?, q)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.last_dim)


def build_SMN(max_turn, maxlen, word_dim, sent_dim, last_dim, num_words, embedding_matrix):

    def MatchByWords(max_turn=10):
        def inside(args):
            u = args[0]  # utterances Tensor
            r = args[1]  # response Tensor
            # concatenate all max_turn (10) M_1 word-word similarity matrices (50x50)
            return Lambda(lambda x: K.stack(x, axis=1))(
                [
                    # dot product by the last axis (embeddings dimension)
                    Dot(axes=(-1, -1))(
                        [(Lambda(lambda x: x[:, turn])(u)), r]
                    ) for turn in range(max_turn)
                ]
            )
        return inside

    def MatchBySegments(sent_dim=200, max_turn=10):
        def inside(args):
            u = args[0]  # utterances Tensor
            r = args[1]  # response Tensor
            return Lambda(lambda x: K.stack(x, axis=1))(
                [
                    # dot product by the last axis (embeddings dimension)
                    Dot(axes=(-1,-1))([
                        Dense(sent_dim, use_bias=False, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer())(Lambda(lambda x: x[:, turn])(u)), r]
                    ) for turn in range(max_turn)
                ]
            )
        return inside


    def DynamicAttention():
        def inside(args):
            """ Last attention layer """
            context_sent_embedding = args[0]
            match = args[1]
            out_dynamic = GRU(last_dim, return_sequences=True, kernel_initializer=tf.orthogonal_initializer())(match)
            # out_dynamic = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\ngru: ', summarize=100))(out_dynamic)
            t = Lambda(lambda x: K.stack(x, axis=1))(
                [
                    Lambda(lambda x: K.tanh(x))(
                        Lambda(lambda x: tf.reduce_sum(x, axis=1))(  # sum over 1st axis, resulting with a vector of shape (q, )
                            Lambda(lambda x: K.stack(x, axis=1))(
                                [
                                    Dense(last_dim, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())(Lambda(lambda x: x[:, i, -1, :])(context_sent_embedding)),
                                    Dense(last_dim, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())(Lambda(lambda x: x[:, i])(out_dynamic))
                                ]
                            )
                        )
                    )
                    for i in range(0, max_turn)
                ]
            )
            # t of shape (?, max_turn, q)
            scores = Dense(1, use_bias=False)(t)
            # scores = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\nscores = ', summarize=10))(scores)
            weights = Softmax(axis=1)(scores)
            # weights = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\nsoftmax scores = ', summarize=10))(weights)
            multiplied = Lambda(lambda op: tf.multiply(op[0], op[1]))([out_dynamic, weights])
            # multiplied = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\nmultipl: ', summarize=100))(
            #     multiplied)
            return Lambda(lambda x: tf.reduce_sum(x, axis=1))(multiplied)  # (?, q)
        return inside

    context_input = Input(shape=(max_turn, maxlen), dtype='int32')
    response_input = Input(shape=(maxlen,), dtype='int32')

    embedding_layer = Embedding(num_words,
                                word_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen
                                )
    sentence2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=tf.orthogonal_initializer())

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    # embedding_layer.trainable = False  # We need to set the param after TimeDistributed is applied

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)
    response_sent_embedding = sentence2vec(response_word_embedding)

    word_match = MatchByWords()([context_word_embedding, response_word_embedding])
    segm_match = MatchBySegments()([context_sent_embedding, response_sent_embedding])
    match_2ch = Lambda(lambda x: K.stack([x[0], x[1]], axis=2))([word_match, segm_match])   # (?, 10, 2, 50, 50) M_1 & M_2 as 2 channels

    conv = TimeDistributed(Conv2D(8, (3, 3), activation='relu', data_format='channels_first', kernel_initializer=tf.contrib.keras.initializers.he_normal()))(match_2ch)
    pool = TimeDistributed(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))(conv)
    flat = TimeDistributed(Flatten())(pool)
    match = TimeDistributed(Dense(last_dim, activation='tanh', kernel_initializer=tf.contrib.layers.xavier_initializer()))(flat)  # v

    ##############################################################################################
    # SMN last
    # output_last = GRU(last_dim, kernel_initializer=tf.orthogonal_initializer())(match)
    # output = Dense(1, activation='sigmoid', kernel_initializer=tf.orthogonal_initializer())(output_last)  # DMN_last
    ##############################################################################################

    ##############################################################################################
    # SMN static
    out_static = GRU(last_dim, return_sequences=True, kernel_initializer=tf.orthogonal_initializer())(match)
    out_static = StaticWeights(last_dim=last_dim, max_turn=max_turn)(out_static)
    output = Dense(1, activation='sigmoid')(out_static)  # DMN_static
    ##############################################################################################

    ##############################################################################################
    # SMN dynamic
    # attention over hidden states h'
    # output_dynamic = DynamicAttention()([context_sent_embedding, match])   # (?, q)
    # output = Dense(1, activation='sigmoid', kernel_initializer=tf.orthogonal_initializer())(output_dynamic)
    ##############################################################################################

    model = Model(inputs=[context_input, response_input], outputs=[output])
    return model
