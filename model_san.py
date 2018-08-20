import numpy as np
np.random.seed(42)

from keras.layers import Conv2D, MaxPooling2D, Embedding, GRU, Reshape
from keras.layers import TimeDistributed
from keras.layers import Dense, Input, Flatten, Lambda, Softmax, Dot
from keras.models import Model, Sequential
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
        self.w = self.add_weight((self.max_turn, 1), name='w', initializer=initializers.glorot_uniform(), trainable=True)
        super(StaticWeights, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(tf.multiply(inputs, self.w), axis=1)  # (?, q)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.last_dim)


def build_SAN(max_turn, maxlen, word_dim, sent_dim, last_dim, num_words, embedding_matrix):

    def AttentionBlock(word_dim=200, sent_dim=200, maxlen=50):
        def inside(args):
            e_u = args[0]  # word-level an utterance representation (?,  50, 200)
            e_r = args[1]  # word-level a response  representation  (?, 50, 200)
            h_u = args[2]  # segment-level an utterance representation (?,  50, 200)
            h_r = args[3]  # segment-level a response  representation  (?, 50, 200)
            t1 = Dense(word_dim, trainable=True, use_bias=True, kernel_initializer=initializers.glorot_uniform())(e_u)
            t1 = Dot(axes=(-1, -1))([e_r, t1])
            t1 = Lambda(lambda x: K.tanh(x))(t1)
            t1 = Dense(maxlen)(t1)    # (?, 50, 50)
            t1 = Softmax(axis=2)(t1)
            broadcasted_e_u = Lambda(lambda x: K.stack(x, axis=1))([e_u for word in range(maxlen)])
            t1 = Lambda(lambda x: tf.multiply(tf.expand_dims(x[0], axis=-1), x[1]))([t1, broadcasted_e_u])  # "weight" the utterance vector according to the response
            t1 = Lambda(lambda x: tf.reduce_sum(x, axis=2))(t1)
            t1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([t1, e_r])  # Hadamard product to the response vector

            t2 = Dense(sent_dim, trainable=True, use_bias=True, kernel_initializer=initializers.glorot_uniform())(h_u)
            t2 = Dot(axes=(-1, -1))([h_r, t2])
            t2 = Lambda(lambda x: K.tanh(x))(t2)
            t2 = Dense(maxlen)(t2)  # (?, 50, 50)
            t2 = Softmax(axis=1)(t2)
            broadcasted_h_u = Lambda(lambda x: K.stack(x, axis=1))([h_u for word in range(maxlen)])
            t2 = Lambda(lambda x: tf.multiply(tf.expand_dims(x[0], axis=-1), x[1]))([t2, broadcasted_h_u])  # "weight" the utterance vector according to the response
            t2 = Lambda(lambda x: tf.reduce_sum(x, axis=2))(t2)
            t2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([t2, h_r])  # Hadamard product to the response vector

            # t = K.concatenate([t1, t2], axis=-1)  # concatenated vector t
            t = Lambda(lambda x: K.stack(x, axis=-1))([t1, t2])
            t = Reshape((maxlen, 2 * sent_dim,))(t)
            return t   # (?, 50, 400)
        return inside

    def WordsAndRepresentationsAttention(max_turn=10):
        def inside(args):
            word_u = args[0]  # word-level utterances Tensor (?, 10, 50, 200)
            word_r = args[1]  # word-level response Tensor   (?, 50, 200)
            segm_u = args[2]  # segment-level utterances Tensor (?, 10, 50, 200)
            segm_r = args[3]  # segment-level utterances Tensor (?, 50, 200)

            T = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [
                            AttentionBlock()([
                                Lambda(lambda x: x[:, turn])(word_u),
                                word_r,
                                Lambda(lambda x: x[:, turn])(segm_u),
                                segm_r
                            ])
                     for turn in range(max_turn)
                ]
            )
            return T
        return inside


    def DynamicAttention():
        def inside(args):
            """ Last attention layer """
            context_sent_embedding = args[0]
            match = args[1]
            out_dynamic = GRU(last_dim, return_sequences=True, kernel_initializer=initializers.orthogonal())(match)
            # out_dynamic = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\ngru: ', summarize=100))(out_dynamic)
            t = Lambda(lambda x: K.stack(x, axis=1))(
                [
                    Lambda(lambda x: K.tanh(x))(
                        Lambda(lambda x: tf.reduce_sum(x, axis=1))(  # sum over 1st axis, resulting with a vector of shape (q, )
                            Lambda(lambda x: K.stack(x, axis=1))(
                                [
                                    Dense(last_dim, use_bias=True, kernel_initializer=initializers.glorot_uniform())(Lambda(lambda x: x[:, i, -1, :])(context_sent_embedding)),
                                    Dense(last_dim, use_bias=True, kernel_initializer=initializers.glorot_uniform())(Lambda(lambda x: x[:, i])(out_dynamic))
                                ]
                            )
                        )
                    )
                    for i in range(0, max_turn)
                ]
            )
            # t of shape (?, max_turn, q)
            scores = Dense(1, use_bias=False, kernel_initializer=initializers.glorot_uniform())(t)
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
    sentence2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=initializers.orthogonal())

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    # embedding_layer.trainable = False  # We need to set the param after TimeDistributed is applied

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)
    response_sent_embedding = sentence2vec(response_word_embedding)

    # Attention Interaction Aggregation
    T = WordsAndRepresentationsAttention()([context_word_embedding, response_word_embedding,
                                                context_sent_embedding, response_sent_embedding])
    gru2 = GRU(last_dim, return_sequences=False, kernel_initializer=initializers.orthogonal())
    v = TimeDistributed(gru2)(T)

    ##############################################################################################
    # SMN last
    output_last = GRU(last_dim, kernel_initializer=initializers.orthogonal())(v)
    output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform())(output_last)  # DMN_last
    ##############################################################################################

    ##############################################################################################
    # SMN static
    # out_static = GRU(last_dim, return_sequences=True, kernel_initializer=initializers.orthogonal())(v)
    # out_static = StaticWeights(last_dim=last_dim, max_turn=max_turn)(out_static)
    # output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform())(out_static)  # DMN_static
    ##############################################################################################

    ##############################################################################################
    # SMN dynamic
    # attention over hidden states h'
    # output_dynamic = DynamicAttention()([context_sent_embedding, v])   # (?, q)
    # output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform())(output_dynamic)
    ##############################################################################################

    model = Model(inputs=[context_input, response_input], outputs=[output])
    return model
