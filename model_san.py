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
    # Debug:
    # max_turn  = 2
    # num_words = 282132
    # word_dim  = 10
    # sent_dim  = 10
    # maxlen    = 5

    def DynamicAttention():
        def inside(args):
            """ Last attention layer """
            context_sent_embedding = args[0]
            match = args[1]
            out_dynamic = GRU(last_dim, return_sequences=True)(match)
            # out_dynamic = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\ngru: ', summarize=100))(out_dynamic)
            t = Lambda(lambda x: K.stack(x, axis=1))(
                [
                    Lambda(lambda x: K.tanh(x))(
                        Lambda(lambda x: tf.reduce_sum(x, axis=1))(  # sum over 1st axis, resulting with a vector of shape (q, )
                            Lambda(lambda x: K.stack(x, axis=1))(
                                [
                                    Dense(last_dim, use_bias=True)(Lambda(lambda x: x[:, i, -1, :])(context_sent_embedding)),
                                    Dense(last_dim, use_bias=True)(Lambda(lambda x: x[:, i])(out_dynamic))
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


    #####################################################################################################
    # Variant 1  some variant... don't know if it works

    def AttentionBlock(word_dim=200, sent_dim=200, maxlen=50):
        def inside(args):
            e_u = args[0]  # word-level an utterance representation (?,  50, 200)
            e_r = args[1]  # word-level a response  representation  (?, 50, 200)
            h_u = args[2]  # segment-level an utterance representation (?,  50, 200)
            h_r = args[3]  # segment-level a response  representation  (?, 50, 200)
            t1 = Dense(word_dim, trainable=True, use_bias=True, kernel_initializer=initializers.glorot_uniform())(e_u)  # (?, 50, 200)
            t1 = Dot(axes=(-1, -1))([e_r, t1])                                          # (?, 50(resp dim), 50(utt dim))
            t1 = Lambda(lambda x: K.tanh(x))(t1)
            t1 = Dense(maxlen, kernel_initializer=initializers.glorot_uniform())(t1)    # (?, 50, 50)
            t1 = Softmax(axis=1)(t1)
            broadcasted_e_u = Lambda(lambda x: K.stack(x, axis=1))([e_u for word in range(maxlen)])
            t1 = Lambda(lambda x: tf.multiply(tf.expand_dims(x[0], axis=-1), x[1]))([t1, broadcasted_e_u])  # "weight" the utterance vector according to the response
            t1 = Lambda(lambda x: tf.reduce_sum(x, axis=1))(t1)
            t1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([t1, e_r])  # Hadamard product to the response vector

            t2 = Dense(sent_dim, trainable=True, use_bias=True, kernel_initializer=initializers.glorot_uniform())(h_u)
            t2 = Dot(axes=(-1, -1))([h_r, t2])
            t2 = Lambda(lambda x: K.tanh(x))(t2)
            t2 = Dense(maxlen, kernel_initializer=initializers.glorot_uniform())(t2)  # (?, 50, 50)
            t2 = Softmax(axis=1)(t2)
            broadcasted_h_u = Lambda(lambda x: K.stack(x, axis=1))([h_u for word in range(maxlen)])
            t2 = Lambda(lambda x: tf.multiply(tf.expand_dims(x[0], axis=-1), x[1]))([t2, broadcasted_h_u])  # "weight" the utterance vector according to the response
            t2 = Lambda(lambda x: tf.reduce_sum(x, axis=1))(t2)
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

            att_block = AttentionBlock()  # 1 attention block
            T = Lambda(lambda x: K.stack(x, axis=1), name="stack_T")(      # for each (utterance-response) pair
                [
                            att_block([
                                Lambda(lambda x: x[:, turn], name="slice_U{}".format(turn))(word_u),
                                word_r,
                                Lambda(lambda x: x[:, turn], name="slice_H{}".format(turn))(segm_u),
                                segm_r
                            ])
                     for turn in range(max_turn)
                ]
            )
            return T
        return inside
    #####################################################################################################

    #####################################################################################################
    # Variant 2 - 'ethalon' implementation - not yet evaluated

    def AttentionBlock2(word_dim, sent_dim, maxlen):
        def inside(args):
            e_u = args[0]  # word-level an utterance representation (?,  50, 200)
            e_ri = args[1]  # word-level representation of an i-th word in a response (?, 200)
            h_u = args[2]  # segment-level an utterance representation (?,  50, 200)
            h_ri = args[3]  # segment-level representation of an i-th segment in a response (?, 200)

            # repr_eu = Dense(word_dim, trainable=True, use_bias=True)(e_u)
            # repr_eri = Dense(word_dim, trainable=True, use_bias=True)(e_ri)
            # broadcasted_e_ri = Lambda(lambda x: K.stack(x, axis=1))([repr_eri for word in range(maxlen)])
            broadcasted_e_ri = Lambda(lambda x: K.stack(x, axis=1))([e_ri for word in range(maxlen)])
            repr_eu = e_u

            # repr_eu = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n1 eu: ', summarize=1000))(
            #     repr_eu)
            # broadcasted_e_ri = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n2 eri: ', summarize=1000))(broadcasted_e_ri)


            m1_i = Lambda(lambda x: K.tanh(x))(
                Lambda(lambda x: tf.reduce_sum(x, axis=1))(
                    Lambda(lambda x: K.stack(x, axis=1))(
                            [repr_eu, broadcasted_e_ri]
                    )
                )

                        # Lambda(lambda x: x[0] + x[1])(  # sum over 1st axis, resulting with a matrix of shape (50, 200, )
                        #         [repr_eu, broadcasted_e_ri]
                        # )
            )
            # m1_i = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n3 m1_i: ', summarize=1000))(
            #     m1_i)

            scores1 = Dense(1, trainable=True, use_bias=False)(m1_i)  # scores

            # scores1 = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n4 scores', summarize=1000))(
            #     scores1)

            weights1 = Softmax(axis=1)(scores1)

            # weights1 = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n5 weights1: ', summarize=1000))(
            #     weights1)

            attended_eu = Lambda(lambda x: K.squeeze(Dot(axes=(1, 1))([x[0], x[1]]), axis=-1))([e_u, weights1])   # (?, 200)

            # attended_eu = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n6 attended_eu: ', summarize=1000))(
            #     attended_eu)

            t1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_eu, e_ri])  # Hadamard product to the response vector

            # t1 = Lambda(lambda x: tf.Print(input_=x, data=[x], message='\n7 t1: ', summarize=1000))(
            #     t1)

            # segments attention
            # repr_hu = Dense(sent_dim, trainable=True, use_bias=True)(h_u)
            # repr_hri = Dense(sent_dim, trainable=True, use_bias=True)(h_ri)
            # broadcasted_h_ri = Lambda(lambda x: K.stack(x, axis=1))([repr_hri for word in range(maxlen)])
            broadcasted_h_ri = Lambda(lambda x: K.stack(x, axis=1))([h_ri for word in range(maxlen)])
            repr_hu = h_u
            m2_i = Lambda(lambda x: K.tanh(x))(
                Lambda(lambda x: tf.reduce_sum(x, axis=1))(
                    Lambda(lambda x: K.stack(x, axis=1))(
                            [repr_hu, broadcasted_h_ri]
                    )
                )
                    # Lambda(lambda x: x[0] + x[1])(  # sum over 1st axis, resulting with a matrix of shape (50, 200, )
                    #     [repr_hu, broadcasted_h_ri]
                    # )
            )
            scores2 = Dense(1, trainable=True, use_bias=False)(m2_i)  # scores
            weights2 = Softmax(axis=1)(scores2)
            attended_hu = Lambda(lambda x: K.squeeze(Dot(axes=(1, 1))([x[0], x[1]]), axis=-1))([h_u, weights2])  # (?, 200)
            t2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_hu, h_ri])  # Hadamard product to the response vector

            # concatenated vector t
            t = Lambda(lambda x: K.stack(x, axis=1))([t1, t2])
            t = Reshape((2 * sent_dim,))(t)
            # t = Dense(sent_dim, trainable=True, use_bias=False)(t) # reduce dimensionality to 200
            return t   # (?, 400)
        return inside

    def WordsAndRepresentationsAttention2(max_turn, maxlen, word_dim, sent_dim):
        def inside(args):
            word_u = args[0]  # word-level utterances Tensor (?, 10, 50, 200)
            word_r = args[1]  # word-level response Tensor   (?, 50, 200)
            segm_u = args[2]  # segment-level utterances Tensor (?, 10, 50, 200)
            segm_r = args[3]  # segment-level utterances Tensor (?, 50, 200)

            # Create hidden representations of U, R and H_u, H_r for attention calculation
            repr_w_u = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [

                    Dense(word_dim, trainable=True, use_bias=True)(Lambda(lambda x: x[:, turn])(word_u))
                    for turn in range(max_turn)
                ]
            )
            repr_s_u = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [

                    Dense(sent_dim, trainable=True, use_bias=True)(Lambda(lambda x: x[:, turn])(segm_u))
                    for turn in range(max_turn)
                ]
            )

            repr_w_r = Lambda(lambda x: K.stack(x, axis=1))(
                        [
                            Dense(word_dim, trainable=True, use_bias=True)(
                                Lambda(lambda x: x[:, resp_word])(word_r)
                            )
                            for resp_word in range(maxlen)
                        ]
                    )
            repr_s_r = Lambda(lambda x: K.stack(x, axis=1))(
                        [
                            Dense(sent_dim, trainable=True, use_bias=True)(
                                Lambda(lambda x: x[:, resp_word])(segm_r)
                            )
                            for resp_word in range(maxlen)
                        ]
                    )

            T = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [
                    Lambda(lambda x: K.stack(x, axis=1))(
                        [
                            AttentionBlock2(maxlen=maxlen, word_dim=word_dim, sent_dim=sent_dim)([
                                Lambda(lambda x: x[:, turn])(repr_w_u),
                                Lambda(lambda x: x[:, resp_word])(repr_w_r),
                                Lambda(lambda x: x[:, turn])(repr_s_u),
                                Lambda(lambda x: x[:, resp_word])(repr_s_r),
                            ])
                            for resp_word in range(maxlen)
                        ]
                    )
                    for turn in range(max_turn)
                ]
            )
            return T  # (?, 10, 50, 400)
        return inside

    #####################################################################################################


    #####################################################################################################
    # Variant 3  same as var 2 but operates with 3D matrices

    def AttentionBlock3(word_dim=200, sent_dim=200, maxlen=50):
        def inside(args):
            e_u = args[0]  # word-level representation of an utterance  (?,  50, 200)
            e_ri = args[1]  # word-level representation of a response (?, 50, 200)
            h_u = args[2]  # segment-level representation of an utterance (?,  50, 200)
            h_ri = args[3]  # segment-level representation of a response (?, 50, 200)

            broadcasted_eu = Lambda(lambda x: K.stack(x, axis=1))([e_u for word in range(maxlen)])  # (?, 50, 50, 200)
            repr_eu = Dense(word_dim, trainable=True, use_bias=True)(broadcasted_eu) # (?, 50, 50, 200)
            broadcasted_e_ri = Lambda(lambda x: K.stack(x, axis=1))([e_ri for word in range(maxlen)])
            repr_eri= Dense(word_dim, trainable=True, use_bias=True)(broadcasted_e_ri)  # (?, 50, 50, 200)
            m1_i = Lambda(lambda x: K.tanh(x))(
                        Lambda(lambda x: tf.reduce_sum(x, axis=1))(  # sum over 1st axis, resulting with a vector of shape (50, 200)
                            Lambda(lambda x: K.stack(x, axis=1))(
                                [repr_eu, repr_eri]
                            )
                        )
            )
            scores1 = Dense(1, trainable=True, use_bias=False)(m1_i)  # scores (?, 50, 50, 1)
            weights1 = Softmax(axis=2)(scores1)
            attended_eu = Lambda(lambda x: K.squeeze(Dot(axes=(2, 2))([x[0], x[1]]), axis=-1))([broadcasted_eu, weights1])   # (?, 50, 50, 200) x (?, 50, 50, 1) = (?, 50, 200)
            t1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_eu, e_ri])  # Hadamard product to the response vector

            # segments attention
            broadcasted_hu = Lambda(lambda x: K.stack(x, axis=1))([e_u for word in range(maxlen)])  # (?, 50, 50, 200)
            repr_hu = Dense(sent_dim, trainable=True, use_bias=True)(broadcasted_hu)
            broadcasted_h_ri = Lambda(lambda x: K.stack(x, axis=1))([h_ri for word in range(maxlen)])
            repr_hri = Dense(sent_dim, trainable=True, use_bias=True)(broadcasted_h_ri)
            m2_i = Lambda(lambda x: K.tanh(x))(
                Lambda(lambda x: tf.reduce_sum(x, axis=1))(
                    # sum over 1st axis, resulting with a vector of shape (200, )
                    Lambda(lambda x: K.stack(x, axis=1))(
                        [repr_hu, repr_hri]
                    )
                )
            )
            scores2 = Dense(1, trainable=True, use_bias=False)(m2_i)  # scores
            weights2 = Softmax(axis=2)(scores2)
            attended_hu = Lambda(lambda x: K.squeeze(Dot(axes=(2, 2))([x[0], x[1]]), axis=-1))([broadcasted_hu, weights2])
            t2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_hu, h_ri])  # Hadamard product to the response vector

            t = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([t1, t2])  # concatenated vector t
            return t   # (?, 50, 400)
        return inside

    def WordsAndRepresentationsAttention3(max_turn=10):
        def inside(args):
            word_u = args[0]  # word-level utterances Tensor (?, 10, 50, 200)
            word_r = args[1]  # word-level response Tensor   (?, 50, 200)
            segm_u = args[2]  # segment-level utterances Tensor (?, 10, 50, 200)
            segm_r = args[3]  # segment-level utterances Tensor (?, 50, 200)

            att_block = AttentionBlock3()  # 1 attention block
            T = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [
                            att_block([
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

    #####################################################################################################


    #####################################################################################################
    # Variant 4 - 'ethalon' implementation - not yet evaluated

    def AttentionBlock4(word_dim=200, sent_dim=200, maxlen=50):
        def inside(args):
            e_u = args[0]  # word-level an utterance representation (?,  50, 200)
            e_ri = args[1]  # word-level representation of an i-th word in a response (?, 200)
            h_u = args[2]  # segment-level an utterance representation (?,  50, 200)
            h_ri = args[3]  # segment-level representation of an i-th segment in a response (?, 200)

            repr_eu = Dense(word_dim, trainable=True, use_bias=True)(e_u)
            repr_eri = Dense(word_dim, trainable=True, use_bias=True)(e_ri)
            broadcasted_e_ri = Lambda(lambda x: K.stack(x, axis=1))([repr_eri for word in range(maxlen)])

            m1_i = Lambda(lambda x: K.tanh(x))(
                        Lambda(lambda x: tf.reduce_sum(x, axis=1))(  # sum over 1st axis, resulting with a matrix of shape (50, 200, )
                            Lambda(lambda x: K.stack(x, axis=1))(
                                [repr_eu, broadcasted_e_ri]
                            )
                        )
            )
            scores1 = Dense(1, trainable=True, use_bias=False)(m1_i)  # scores
            weights1 = Softmax(axis=1)(scores1)
            attended_eu = Lambda(lambda x: K.squeeze(Dot(axes=(1, 1))([x[0], x[1]]), axis=-1))([e_u, weights1])   # (?, 200)
            t1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_eu, e_ri])  # Hadamard product to the response vector

            # segments attention
            repr_hu = Dense(sent_dim, trainable=True, use_bias=True)(h_u)
            repr_hri = Dense(sent_dim, trainable=True, use_bias=True)(h_ri)
            broadcasted_h_ri = Lambda(lambda x: K.stack(x, axis=1))([repr_hri for word in range(maxlen)])
            m2_i = Lambda(lambda x: K.tanh(x))(
                Lambda(lambda x: tf.reduce_sum(x, axis=1))(
                    # sum over 1st axis, resulting with a matrix of shape (50, 200, )
                    Lambda(lambda x: K.stack(x, axis=1))(
                        [repr_hu, broadcasted_h_ri]
                    )
                )
            )
            scores2 = Dense(1, trainable=True, use_bias=False)(m2_i)  # scores
            weights2 = Softmax(axis=1)(scores2)
            attended_hu = Lambda(lambda x: K.squeeze(Dot(axes=(1, 1))([x[0], x[1]]), axis=-1))([h_u, weights2])  # (?, 200)
            t2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([attended_hu, h_ri])  # Hadamard product to the response vector

            # concatenated vector t
            t = Lambda(lambda x: K.stack(x, axis=1))([t1, t2])
            t = Reshape((2 * sent_dim,))(t)
            # t = Dense(sent_dim, trainable=True, use_bias=False)(t) # reduce dimensionality to 200
            return t   # (?, 400)
        return inside

    def WordsAndRepresentationsAttention4(max_turn=10, maxlen=50):
        def inside(args):
            word_u = args[0]  # word-level utterances Tensor (?, 10, 50, 200)
            word_r = args[1]  # word-level response Tensor   (?, 50, 200)
            segm_u = args[2]  # segment-level utterances Tensor (?, 10, 50, 200)
            segm_r = args[3]  # segment-level utterances Tensor (?, 50, 200)
            T = Lambda(lambda x: K.stack(x, axis=1))(      # for each (utterance-response) pair
                [
                    Lambda(lambda x: K.stack(x, axis=1))(
                        [
                            AttentionBlock2()([
                                Lambda(lambda x: x[:, turn])(word_u),
                                Lambda(lambda x: x[:, resp_word])(word_r),
                                Lambda(lambda x: x[:, turn])(segm_u),
                                Lambda(lambda x: x[:, resp_word])(segm_r),
                            ])
                            for resp_word in range(maxlen)
                        ]
                    )
                    for turn in range(max_turn)
                ]
            )
            return T  # (?, 10, 50, 400)
        return inside

    #####################################################################################################

    # Model
    ci = Input(shape=(10, 50), dtype='int32')
    ri = Input(shape=(50,), dtype='int32')

    # context_input = Lambda(lambda x: x[:, :max_turn, :maxlen])(ci)
    # response_input = Lambda(lambda x: x[:, :maxlen])(ri)

    embedding_layer = Embedding(num_words,
                                word_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen
                                )
    sentence2vec = GRU(sent_dim, return_sequences=True)

    context_word_embedding = TimeDistributed(embedding_layer, name="U")(ci)
    response_word_embedding = embedding_layer(ri)

    # embedding_layer.trainable = False  # We need to set the param after TimeDistributed is applied

    context_sent_embedding = TimeDistributed(sentence2vec, name="H")(context_word_embedding)
    response_sent_embedding = sentence2vec(response_word_embedding)

    # Attention Interaction Aggregation
    T = WordsAndRepresentationsAttention2(max_turn=max_turn, maxlen=maxlen, word_dim=word_dim, sent_dim=sent_dim)\
        ([context_word_embedding, response_word_embedding,
          context_sent_embedding, response_sent_embedding])

    # Get hidden vectors v = [v1, ..., v10]
    gru_v = GRU(400, return_sequences=False)  # may be use 200 instead of 400
    v = TimeDistributed(gru_v)(T)  # T: (?, 10, 50, 400) -> v: (?, 10, 400{take last hidden})

    ##############################################################################################
    # SMN last
    output_last = GRU(last_dim, return_sequences=False)(v)
    output = Dense(1, activation='sigmoid')(output_last)  # DMN_last
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

    model = Model(inputs=[ci, ri], outputs=[output])
    return model
