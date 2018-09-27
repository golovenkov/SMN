import argparse
import numpy as np

np.random.seed(42)

import joblib
import tensorflow as tf

tf.set_random_seed(42)

import tensorflow_hub as hub

class SCN_ELMo():
    def __init__(self,
                 embedding_matrix=None,
                 train_data_context=None,
                 train_data_context_len=None,
                 train_data_response=None,
                 train_data_response_len=None,
                 valid_data_context=None,
                 valid_data_context_len=None,
                 valid_data_response=None,
                 valid_data_response_len=None,
                 test_data_context=None,
                 test_data_context_len=None,
                 test_data_response=None,
                 test_data_response_len=None,
                 train_labels=None,
                 num_words=200000,
                 batch_size=500,
                 valid_batch_size=200,
                 version=2,
                 trainable=False
                 ):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 1024
        self.rnn_units = 200
        # self.total_words = 434511
        self.total_words = num_words + 1
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        self.version = version
        self.embedding_matrix = embedding_matrix

        self.train_data_context = train_data_context
        self.train_data_context_len = train_data_context_len
        self.train_data_response = train_data_response
        self.train_data_response_len = train_data_response_len

        self.valid_data_context = valid_data_context
        self.valid_data_context_len = valid_data_context_len
        self.valid_data_response = valid_data_response
        self.valid_data_response_len = valid_data_response_len

        # initialize test data if it exists
        self.test_data_context = test_data_context
        self.test_data_context_len = test_data_context_len
        self.test_data_response = test_data_response
        self.test_data_response_len = test_data_response_len

        self.train_labels = train_labels
        # self.validation_step = len(self.train_data_context) // 2
        self.validation_step = len(self.train_data_context)

        self.trainable = trainable

    def BuildModel(self):
        # labels
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        # context
        self.utterance_ph = tf.placeholder(tf.string, shape=(None, self.max_num_utterance, self.max_sentence_len))  # tf.string
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        # response
        self.response_ph = tf.placeholder(tf.string, shape=(None, self.max_sentence_len))  # tf.string
        self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))

        # self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        # word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
        #                                                               word_embedding_size), dtype=tf.float32, trainable=self.trainable)
        # self.embedding_init = word_embeddings.assign(self.embedding_ph)
        # all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        # response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)

        # all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1) # list of 10 tensors (?, 50, 200)
        # all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)

        # ELMo
        elmo_embeddings = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False, name="ELMo")
        all_utterance = tf.unstack(self.utterance_ph, num=self.max_num_utterance, axis=1)  # list of 10 tensors (?, 50)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)  # list of 10 tensors (?, )
        all_utterance_embeddings = []
        # embed context
        for utterance, utterance_len in zip(all_utterance, all_utterance_len):
            elmo_emb = elmo_embeddings(
                inputs={"tokens": utterance,
                        "sequence_len": utterance_len
                        },
                signature="tokens",
                as_dict=True)['elmo']
            all_utterance_embeddings.append(elmo_emb)
        # embed response
        response_embeddings = elmo_embeddings(
                inputs={"tokens": self.response_ph,
                        "sequence_len": self.response_len_ph
                        },
                signature="tokens",
                as_dict=True)['elmo']

        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                       response_embeddings,
                                                       sequence_length=self.response_len_ph,
                                                       dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                            utterance_embeddings,
                                                            sequence_length=utterance_len,
                                                            dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.reshape(pooling_layer, [-1, 16*16*8]), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU,
                                           tf.stack(matching_vectors, axis=0, name='matching_stack'),  # resulting shape: (10, ?, 50)
                                           dtype=tf.float32,
                                           time_major=True,
                                           scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)

    def restore_from_epoch(self, epoch_n):
        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess=sess, save_path="model/model.{}".format(epoch_n))
        return sess

    def create_indices(self, data_length, shuffle=True):
        indices_10 = []
        indices_100 = []
        idx = 0
        shift = 100
        epoch = 0
        indices = [i for i in range(1, 100)]
        while epoch < data_length // shift:
            shuffled_indices = [i for i in range(1, 100)]
            if shuffle:
                np.random.shuffle(shuffled_indices)
            indices_10.append(idx)
            indices_10.extend([i + shift * epoch for i in shuffled_indices[:9]])  # shuffle for r10@k
            indices_100.append(idx)
            indices_100.extend([i + shift * epoch for i in indices])  # do not shuffle for r100@k
            idx += shift
            epoch += 1

        return indices_10, indices_100

    def train(self, countinue_train=False, previous_modelpath="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('SCN_Graph', sess.graph)
            labels = self.train_labels
            history, history_len = self.train_data_context, self.train_data_context_len
            response, response_len = self.train_data_response, self.train_data_response_len
            # actions, actions_len = np.array(self.train_data['response']), np.array(self.train_data['response_len'])
            # actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            # history, history_len = np.array(history), np.array(history_len)
            if countinue_train == False:
                sess.run(init)
                # sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding_matrix})  # TODO: init existing elmo ?
            else:
                saver.restore(sess, previous_modelpath)
            low = 0
            epoch = 1
            indices_10, indices_100 = self.create_indices(data_length=len(self.valid_data_context),
                                                          shuffle=True
                                                          )  # valid indices
            print('\nshuffle train data, len: {}'.format(len(history)))
            idx = np.array([i for i in range(len(history))])
            np.random.shuffle(idx)
            print('epoch', epoch)
            while epoch <= 15:
                curr_batch_size = min(low + self.batch_size, len(history)) - low
                h_ = []
                r_ = []
                h_l = []
                r_l = []
                l = []
                ind_ = idx[low:low + curr_batch_size]
                for i in ind_:
                    h_.append(history[i])
                    r_.append(response[i])
                    h_l.append(history_len[i])
                    r_l.append(response_len[i])
                    l.append(labels[i])

                feed_dict = {
                    self.utterance_ph: h_,
                    self.all_utterance_len_ph: h_l,
                    self.response_ph: r_,
                    self.response_len_ph: r_l,
                    self.y_true: l
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch * len(idx) + low)
                low += curr_batch_size
                if low % (len(idx)//10) == 0:
                    print("epoch: {} iter: {} train_loss: {:g}".format(epoch, low, sess.run(self.total_loss, feed_dict=feed_dict)))
                if low % self.validation_step == 0:
                    print("Validation:")
                    if self.version == 1:
                        self.evaluate_from_n_utterances(sess)
                        print("Test:")
                        self.test(sess)
                    elif self.version == 2:
                        self.evaluate_from_n_utterances(sess)
                        print('Test:')
                        self.test(sess)
                    elif self.version == 3:
                        # only evaluate for v3 because we don't have test data yet!
                        self.evaluate_from_n_utterances(sess, n_utt=100, indices=indices_100)
                        self.evaluate_from_n_utterances(sess, n_utt=10, indices=indices_10)

                if low >= len(history):
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    epoch += 1

                    print('\nepoch={i}'.format(i=epoch))
                    print('reshuffle train data')
                    idx = np.array([i for i in range(len(history))])
                    np.random.shuffle(idx)

    def get_dims_and_recall(self, length, k=10):
        return length//k, k, [1, 2, 5] if k == 10 else [1, 2, 5, 10, 50]

    def evaluate_from_n_utterances(self, sess, n_utt=10, indices=None):
        self.all_candidate_scores = []
        history, history_len = self.valid_data_context, self.valid_data_context_len
        response, response_len = self.valid_data_response, self.valid_data_response_len
        low = 0

        if self.version in [1, 2]:
            indices = [i for i in range(len(history))]  # no need to shuffle indices r10@k for v1
        while True:
            h_ = []
            r_ = []
            h_l = []
            r_l = []
            ind_ = indices[low:low + self.valid_batch_size]
            for i in ind_:
                h_.append(history[i])
                r_.append(response[i])
                h_l.append(history_len[i])
                r_l.append(response_len[i])
            
            feed_dict = {self.utterance_ph: h_,
                         self.all_utterance_len_ph: h_l,
                         self.response_ph: r_,
                         self.response_len_ph: r_l
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= len(indices):
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices))
        elif self.version == 3:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices), k=n_utt)

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def test(self, sess):
        self.all_candidate_scores = []
        history, history_len = self.test_data_context, self.test_data_context_len
        response, response_len = self.test_data_response, self.test_data_response_len
        low = 0
        while True:
            feed_dict = {self.utterance_ph: history[low:low + self.valid_batch_size],
                         self.all_utterance_len_ph: history_len[low:low + self.valid_batch_size],
                         self.response_ph: response[low:low + self.valid_batch_size],
                         self.response_len_ph: response_len[low:low + self.valid_batch_size]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= len(history):
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(history))
        elif self.version == 3:
            pass  # no test data for DSTC7 yet!

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def evaluate_recall(self, y, k=1):
        num_examples = float(len(y))
        num_correct = 0

        for predictions in y:
            if 0 in predictions[:k]:
                num_correct += 1
        return num_correct / num_examples

class SCN():
    def __init__(self,
                 embedding_matrix=None,
                 train_data_context=None,
                 train_data_context_len=None,
                 train_data_response=None,
                 train_data_response_len=None,
                 valid_data_context=None,
                 valid_data_context_len=None,
                 valid_data_response=None,
                 valid_data_response_len=None,
                 test_data_context=None,
                 test_data_context_len=None,
                 test_data_response=None,
                 test_data_response_len=None,
                 train_labels=None,
                 num_words=200000,
                 batch_size=500,
                 valid_batch_size=200,
                 version=1,
                 trainable=False
                 ):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_units = 200
        # self.total_words = 434511
        self.total_words = num_words + 1
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        self.version = version
        self.embedding_matrix = embedding_matrix

        self.train_data_context = train_data_context
        self.train_data_context_len = train_data_context_len
        self.train_data_response = train_data_response
        self.train_data_response_len = train_data_response_len

        self.valid_data_context = valid_data_context
        self.valid_data_context_len = valid_data_context_len
        self.valid_data_response = valid_data_response
        self.valid_data_response_len = valid_data_response_len

        # initialize test data if it exists
        self.test_data_context = test_data_context
        self.test_data_context_len = test_data_context_len
        self.test_data_response = test_data_response
        self.test_data_response_len = test_data_response_len

        self.train_labels = train_labels
        # self.validation_step = len(self.train_data_context) // 2
        self.validation_step = len(self.train_data_context)

        self.trainable = trainable

    def BuildModel(self):
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                      word_embedding_size), dtype=tf.float32, trainable=self.trainable)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1) # list of 10 tensors (?, 200)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                       response_embeddings,
                                                       sequence_length=self.response_len_ph,
                                                       dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                            utterance_embeddings,
                                                            sequence_length=utterance_len,
                                                            dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU,
                                           tf.stack(matching_vectors, axis=0, name='matching_stack'),  # resulting shape: (10, ?, 50)
                                           dtype=tf.float32,
                                           time_major=True,
                                           scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
        self.train_op = optimizer.minimize(self.total_loss)

    def restore_from_epoch(self, epoch_n):
        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess=sess, save_path="model/model.{}".format(epoch_n))
        return sess

    def create_indices(self, data_length, shuffle=True):
        indices_10 = []
        indices_100 = []
        idx = 0
        shift = 100
        epoch = 0
        indices = [i for i in range(1, 100)]
        while epoch < data_length // shift:
            shuffled_indices = [i for i in range(1, 100)]
            if shuffle:
                np.random.shuffle(shuffled_indices)
            indices_10.append(idx)
            indices_10.extend([i + shift * epoch for i in shuffled_indices[:9]])  # shuffle for r10@k
            indices_100.append(idx)
            indices_100.extend([i + shift * epoch for i in indices])  # do not shuffle for r100@k
            idx += shift
            epoch += 1

        return indices_10, indices_100

    def train(self, countinue_train=False, previous_modelpath="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('SCN_Graph', sess.graph)
            labels = self.train_labels
            history, history_len = self.train_data_context, self.train_data_context_len
            response, response_len = self.train_data_response, self.train_data_response_len
            # actions, actions_len = np.array(self.train_data['response']), np.array(self.train_data['response_len'])
            # actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            # history, history_len = np.array(history), np.array(history_len)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding_matrix})
            else:
                saver.restore(sess, previous_modelpath)
            low = 0
            epoch = 1
            indices_10, indices_100 = self.create_indices(data_length=len(self.valid_data_context),
                                                          shuffle=True
                                                          )  # valid indices
            print('\nshuffle train data, len: {}'.format(len(history)))
            idx = np.array([i for i in range(len(history))])
            np.random.shuffle(idx)
            print('epoch', epoch)
            while epoch <= 15:
                curr_batch_size = min(low + self.batch_size, history.shape[0]) - low
                feed_dict = {
                    self.utterance_ph: history[idx[low:low + curr_batch_size]],
                    self.all_utterance_len_ph: history_len[idx[low:low + curr_batch_size]],
                    self.response_ph: response[idx[low:low + curr_batch_size]],
                    self.response_len_ph: response_len[idx[low:low + curr_batch_size]],
                    self.y_true: labels[idx[low: low + curr_batch_size]]
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch * len(idx) + low)
                low += curr_batch_size
                if low % (len(idx)//10) == 0:
                    print("epoch: {} iter: {} train_loss: {:g}".format(epoch, low, sess.run(self.total_loss, feed_dict=feed_dict)))
                if low % self.validation_step == 0:
                    print("Validation:")
                    if self.version == 1:
                        self.evaluate_from_n_utterances(sess)
                        print("Test:")
                        self.test(sess)
                    elif self.version == 2:
                        self.evaluate_from_n_utterances(sess)
                        print('Test:')
                        self.test(sess)
                    elif self.version == 3:
                        # only evaluate for v3 because we don't have test data yet!
                        self.evaluate_from_n_utterances(sess, n_utt=100, indices=indices_100)
                        self.evaluate_from_n_utterances(sess, n_utt=10, indices=indices_10)

                if low >= history.shape[0]:
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    epoch += 1

                    print('\nepoch={i}'.format(i=epoch))
                    print('reshuffle train data')
                    idx = np.array([i for i in range(len(history))])
                    np.random.shuffle(idx)

    def get_dims_and_recall(self, length, k=10):
        return length//k, k, [1, 2, 5] if k == 10 else [1, 2, 5, 10, 50]

    def evaluate_from_n_utterances(self, sess, n_utt=10, indices=None):
        self.all_candidate_scores = []
        history, history_len = self.valid_data_context, self.valid_data_context_len
        response, response_len = self.valid_data_response, self.valid_data_response_len
        low = 0

        if self.version in [1, 2]:
            indices = [i for i in range(len(history))]  # no need to shuffle indices r10@k for v1
        while True:
            feed_dict = {self.utterance_ph: history[indices[low:low + self.valid_batch_size]],
                         self.all_utterance_len_ph: history_len[indices[low:low + self.valid_batch_size]],
                         self.response_ph: response[indices[low:low + self.valid_batch_size]],
                         self.response_len_ph: response_len[indices[low:low + self.valid_batch_size]]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= len(indices):
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices))
        elif self.version == 3:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices), k=n_utt)

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def test(self, sess):
        self.all_candidate_scores = []
        history, history_len = self.test_data_context, self.test_data_context_len
        response, response_len = self.test_data_response, self.test_data_response_len
        low = 0
        while True:
            feed_dict = {self.utterance_ph: history[low:low + self.valid_batch_size],
                         self.all_utterance_len_ph: history_len[low:low + self.valid_batch_size],
                         self.response_ph: response[low:low + self.valid_batch_size],
                         self.response_len_ph: response_len[low:low + self.valid_batch_size]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(history))
        elif self.version == 3:
            pass  # no test data for DSTC7 yet!

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def evaluate_recall(self, y, k=1):
        num_examples = float(len(y))
        num_correct = 0

        for predictions in y:
            if 0 in predictions[:k]:
                num_correct += 1
        return num_correct / num_examples

from tensorflow.contrib.layers import layer_norm
from tensorflow.contrib.layers import fully_connected

def variable_summaries(var, parent_scope=None):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  if parent_scope is None:
      parent_scope = ""
  with tf.name_scope(parent_scope+'summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', dtype=tf.float32, initializer=tf.zeros(params_shape), trainable=True)
        gamma = tf.get_variable('gamma', dtype=tf.float32, initializer=tf.ones(params_shape), trainable=True)
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs, gamma, beta

from collections import defaultdict
def print_number_of_parameters():
    """
    Print number of *trainable* parameters in the network
    """
    print('Number of parameters: ')
    variables = tf.trainable_variables()
    blocks = defaultdict(int)
    for var in variables:
        # Get the top level scope name of variable
        block_name = var.name.split('/')[0]
        number_of_parameters = np.prod(var.get_shape().as_list())
        blocks[block_name] += number_of_parameters
    for block_name, cnt in blocks.items():
        print("{} - {}.".format(block_name, cnt))
    total_num_parameters = np.sum(list(blocks.values()))
    print('Total number of parameters equal {}'.format(total_num_parameters))

from utils import layers
from utils import operations as op


class DeepAttentionMatchingNetwork_old():
    def __init__(self,
                 embedding_matrix=None,
                 train_data_context=None,
                 train_data_context_len=None,
                 train_data_response=None,
                 train_data_response_len=None,
                 valid_data_context=None,
                 valid_data_context_len=None,
                 valid_data_response=None,
                 valid_data_response_len=None,
                 test_data_context=None,
                 test_data_context_len=None,
                 test_data_response=None,
                 test_data_response_len=None,
                 train_labels=None,
                 num_words=200000,
                 batch_size=500,
                 valid_batch_size=200,
                 version=1,
                 trainable=False
                 ):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_units = 200
        # self.total_words = 434511
        self.total_words = num_words + 1
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        self.version = version
        self.embedding_matrix = embedding_matrix

        self.train_data_context = train_data_context
        self.train_data_context_len = train_data_context_len
        self.train_data_response = train_data_response
        self.train_data_response_len = train_data_response_len

        self.valid_data_context = valid_data_context
        self.valid_data_context_len = valid_data_context_len
        self.valid_data_response = valid_data_response
        self.valid_data_response_len = valid_data_response_len

        # initialize test data if it exists
        self.test_data_context = test_data_context
        self.test_data_context_len = test_data_context_len
        self.test_data_response = test_data_response
        self.test_data_response_len = test_data_response_len

        self.train_labels = train_labels
        self.validation_step = len(self.train_data_context)

        self.trainable = trainable
        self.stack_num = 3

    def AttentiveModule(self, query, key, value, reuse=None, scope_name=None, inp=""):
        transposed_key = tf.transpose(key, perm=[0, 2, 1], name='transposed_key')  # (?, 50, 200) -> (?, 200, 50)
        scores = tf.matmul(query, transposed_key, name='att_matmul')  # (?, 50, 200) x (?, 200, 50) = (?, 50, 50)
        scores = tf.divide(scores, tf.sqrt(tf.cast(self.word_embedding_size, dtype=tf.float32)), name='att_div')
        # scores = tf.Print(scores, [query[0], transposed_key[0], scores[0]],
        #                           inp + " " + scope_name + "/query, key & mul(*): ", summarize=3)
        att = tf.nn.softmax(scores, axis=2, name='att_softmax')  # axis=2: by columns  Resulting shape: (?, 50, 50)

        attended_value = tf.matmul(att, value, name='attended_value')  # (?, 50, 50) x (?, 50, 200) = (?, 50, 200)
        # attended_value = tf.Print(attended_value, [attended_value[0]], inp + " " + scope_name + "/att_matmux: ", summarize=10)

        x = tf.add(attended_value, query, name='att_add1')
        x = layer_norm(x, reuse=reuse, scope=scope_name + '/l_norm1')  # begin_params_axis=1,
        # x, g, b = normalize(x, scope=scope_name+'/norm1', reuse=reuse)

        # x = tf.Print(x, [x[0]], inp + " " + scope_name + "/norm1: ", summarize=10)

        ffn1 = fully_connected(x, self.word_embedding_size, activation_fn=tf.nn.relu,
                               weights_initializer=tf.keras.initializers.he_normal(),
                               reuse=reuse, scope=scope_name + "/ffn1")
        # ffn1 = tf.Print(ffn1, [ffn1[0]], inp + " " + scope_name + "/ffn1: ", summarize=10000)
        ffn = fully_connected(ffn1, self.word_embedding_size, activation_fn=None,
                              # weights_initializer=tf.initializers.lecun_normal(),  # 1/sqrt(n)
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              reuse=reuse, scope=scope_name + '/ffn2')
        # ffn = tf.Print(ffn, [ffn[0]], inp + " " + scope_name + "/ffn: ", summarize=10000)

        # x = tf.Print(x, [x[0], b], inp + " " + scope_name + "/x, gamma, beta: ", summarize=10)
        # with tf.variable_scope(scope_name, reuse=reuse):
        #     # w1 = tf.get_variable('filter_layer_w1', shape=(self.word_embedding_size, self.word_embedding_size),
        #     #                      initializer=tf.initializers.he_normal(), trainable=True, dtype=tf.float32)
        #     # b1 = tf.get_variable('filter_layer_b1', shape=(self.word_embedding_size,),
        #     #                      initializer=tf.initializers.zeros(), trainable=True, dtype=tf.float32,
        #     #                      # regularizer=tf.keras.regularizers.l2()
        #     #                      )
        #     # ffn = tf.nn.relu(tf.add(tf.tensordot(x, w1, axes=1), b1))
        #     ffn = tf.layers.Dense(self.word_embedding_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")(x)
        #     # ffn = tf.Print(ffn, [ffn[0], b1], inp + " " + scope_name + "/ffn1, w1, b1: ", summarize=10)
        #
        #     # w2 = tf.get_variable('output_layer_w2', shape=(self.word_embedding_size, self.word_embedding_size),
        #     #                      initializer=tf.initializers.glorot_uniform(), trainable=True, dtype=tf.float32)
        #     # b2 = tf.get_variable('output_layer_b2', shape=(self.word_embedding_size, ),
        #     #                      initializer=tf.initializers.zeros(), trainable=True, dtype=tf.float32,
        #     #                      # regularizer=tf.keras.regularizers.l2()
        #     #                      )
        #     # ffn = tf.add(tf.tensordot(ffn, w2, axes=1), b2)
        #     ffn = tf.layers.Dense(self.word_embedding_size, use_bias=True, name="output_layer")(ffn)
        #     # ffn = tf.Print(ffn, [ffn[0], b2], inp + " " + scope_name + "/ffn2, w2, b2: ", summarize=10)

        # print weights of FFN
        # print(tf.global_variables())
        # with tf.variable_scope(scope_name+"/filter_layer", reuse=reuse):
        #     w1 = tf.get_variable('kernel')
        #     b1 = tf.get_variable('bias')
        # with tf.variable_scope(scope_name+'/output_layer', reuse=reuse):
        #     w2 = tf.get_variable('kernel')
        #     b2 = tf.get_variable('bias')
        #     ffn = tf.Print(ffn, [w1, b1, w2, b2], "w1, b1, w2, b2: ", summarize=100000)

        # with tf.variable_scope(scope_name, reuse=reuse):
        #     # Inner layer
        #     params = {"inputs": x, "filters": 200, "kernel_size": 1,
        #               "activation": tf.nn.relu, "use_bias": True}
        #     outputs = tf.layers.conv1d(**params)
        #     outputs = tf.Print(outputs, [outputs[0]], inp + " " + scope_name + "/ffn_1: ", summarize=100000)
        #     # Readout layer
        #     params = {"inputs": outputs, "filters": 200, "kernel_size": 1,
        #               "activation": None, "use_bias": True}
        #     ffn = tf.layers.conv1d(**params)
        #     ffn = tf.Print(ffn, [ffn[0]], inp + " " + scope_name + "/ffn_2: ", summarize=100000)

        # # Residual connection
        # x += ffn

        x = tf.add(x, ffn, name='att_add2')
        # x, _, _ = normalize(x, scope=scope_name+'/norm2', reuse=reuse)
        x = layer_norm(x, reuse=reuse, scope=scope_name + '/l_norm2')  # (?, 50, 200)
        return x

    def BuildModel(self):
        with tf.variable_scope('inputs'):
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.y_true = tf.placeholder(tf.int32, shape=(None,))
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        with tf.variable_scope('embedding_matrix_init'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                          word_embedding_size), dtype=tf.float32,
                                              trainable=self.trainable)
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
        with tf.variable_scope('embedding_lookup'):
            all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
            response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        with tf.variable_scope('context_unstack'):
            all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance,
                                                  axis=1)  # list of 10 tensors (?, 200)

        U = []
        U.append(all_utterance_embeddings)  # U: [[(?, 50, 200), (?, 50, 200), ... 10 times], ...]
        R = []
        R.append(response_embeddings)  # R: [(?, 50, 200), ...]
        M_self = []
        M_cross = []

        reuse = None
        scope_name = 'L'  # .format(L)
        for L in range(0, self.stack_num):
            # scope_name = 'L{}'.format(L); reuse = None

            r = R[L]
            resp_transposed = tf.transpose(r, perm=[0, 2, 1], name='resp_transposed')
            self_matching_matrices = []
            cross_matching_matrices = []

            U_i = []
            response_representation = self.AttentiveModule(r, r, r, reuse=reuse, scope_name=scope_name, inp="r,r,r")
            # variable_summaries(response_representation, scope_name+'/r_repr/')
            if reuse is None:
                reuse = True

            for index, u in enumerate(U[L]):  # For each (u, r) pair
                # 3.4 Utterance and response representation
                u_r = self.AttentiveModule(u, u, u, reuse=reuse, scope_name=scope_name,
                                               inp="u{0},u{0},u{0}".format(index))
                # variable_summaries(u_r, scope_name+'/u_repr/')
                U_i.append(u_r)

                # self-attention-match
                self_dot = tf.matmul(u, resp_transposed, name='matmul_self')
                self_matching_matrices.append(self_dot)

                # ############################
                # cross-attention match
                u_hat = self.AttentiveModule(u, r, r, reuse=reuse, scope_name=scope_name,
                                                 inp="u{0},r{0},r{0}".format(index))
                # variable_summaries(u_hat, scope_name+'/u_hat/')
                r_hat = self.AttentiveModule(r, u, u, reuse=reuse, scope_name=scope_name,
                                                 inp="r{0},u{0},u{0}".format(index))
                # variable_summaries(r_hat, scope_name+'/r_hat/')

                cross_dot = tf.matmul(
                    u_hat,
                    tf.transpose(r_hat, perm=[0, 2, 1], name='r_hat_transp'), name='matmul_cross'
                )
                cross_matching_matrices.append(cross_dot)

                # ############################
            U.append(U_i)  # next L : (L + 1)
            R.append(response_representation)  # next L : (L + 1)

            self_stacked = tf.stack(self_matching_matrices, axis=1, name=scope_name + '/self_matrix')
            variable_summaries(self_stacked, scope_name + '/self_matrix/')
            cross_stacked = tf.stack(cross_matching_matrices, axis=1, name=scope_name + '/cross_matrix')
            variable_summaries(cross_stacked, scope_name + '/cross_matrix/')

            M_self.append(self_stacked)  # add (?, 10, 50, 50)
            M_cross.append(cross_stacked)  # add (?, 10, 50, 50)

        M = []
        for m1, m2 in zip(M_self, M_cross):
            M.append(m1)
            M.append(m2)
        Q = tf.stack(M, axis=1, name='Q')
        # Q = tf.concat([tf.stack([m for m in M_self], axis=1), tf.stack([m for m in M_cross], axis=1)], axis=1, name='HyperQub')
        conv_layer1 = tf.layers.conv3d(Q, filters=32, kernel_size=(3, 3, 3), padding='SAME',
                                       data_format='channels_first',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       activation=tf.nn.relu, name='conv1')
        # variable_summaries(conv_layer1, 'conv1/')
        pooling_layer1 = tf.layers.max_pooling3d(conv_layer1, (3, 3, 3), strides=(3, 3, 3),
                                                 padding='SAME',
                                                 data_format='channels_first',
                                                 name='max_pool1')
        # variable_summaries(pooling_layer1, 'pool1/')
        conv_layer2 = tf.layers.conv3d(pooling_layer1, filters=16, kernel_size=(3, 3, 3), padding='SAME',
                                       data_format='channels_first',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       activation=tf.nn.relu, name='conv2')
        # variable_summaries(conv_layer2, 'conv2/')
        pooling_layer2 = tf.layers.max_pooling3d(conv_layer2, (3, 3, 3), strides=(3, 3, 3),
                                                 padding='SAME',
                                                 data_format='channels_first',
                                                 name='max_pool2')
        # variable_summaries(conv_layer2, 'pool2/')
        last_vector = tf.contrib.layers.flatten(pooling_layer2, "flatten")  # (?, 400)
        variable_summaries(last_vector, 'flatten/')
        logits = tf.layers.dense(last_vector, 2,
                                 # initializer: according https://arxiv.org/pdf/1704.08863.pdf (Section 3.2)
                                 # kernel_initializer=tf.initializers.variance_scaling(
                                 #     scale=3.6**2, mode="fan_in", distribution="truncated_normal")
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                 , name='final_v')
        self.y_pred = tf.nn.softmax(logits, name="y_pred")
        self.total_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
            gradients = optimizer.compute_gradients(self.total_loss)
            clipped_gradients = [(tf.clip_by_value(grad, -1e17, 1e17), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(clipped_gradients)

        # print num of params
        print(Q.shape)  # (?, 2(L+1), 10, 50, 50)
        print(last_vector.shape)
        print_number_of_parameters()

    def restore_from_epoch(self, epoch_n):
        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess=sess, save_path="model/model.{}".format(epoch_n))
        return sess

    def create_indices(self, data_length, shuffle=True):
        indices_10 = []
        indices_100 = []
        idx = 0
        shift = 100
        epoch = 0
        indices = [i for i in range(1, 100)]
        while epoch < data_length // shift:
            shuffled_indices = [i for i in range(1, 100)]
            if shuffle:
                np.random.shuffle(shuffled_indices)
            indices_10.append(idx)
            indices_10.extend([i + shift * epoch for i in shuffled_indices[:9]])  # shuffle for r10@k
            indices_100.append(idx)
            indices_100.extend([i + shift * epoch for i in indices])  # do not shuffle for r100@k
            idx += shift
            epoch += 1

        return indices_10, indices_100

    def train(self, countinue_train=False, previous_modelpath="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('DAM_Graph', sess.graph)
            labels = self.train_labels
            history, history_len = self.train_data_context, self.train_data_context_len
            response, response_len = self.train_data_response, self.train_data_response_len

            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding_matrix})
            else:
                saver.restore(sess, previous_modelpath)
            low = 0
            epoch = 1
            indices_10, indices_100 = self.create_indices(data_length=len(self.valid_data_context),
                                                          shuffle=True
                                                          )  # valid indices
            print('\nshuffle train data, len: {}'.format(len(history)))
            idx = np.array([i for i in range(len(history))])
            np.random.shuffle(idx)
            print('epoch', epoch)
            while epoch <= 15:
                curr_batch_size = min(low + self.batch_size, history.shape[0]) - low
                feed_dict = {
                    self.utterance_ph: history[idx[low:low + curr_batch_size]],
                    # self.all_utterance_len_ph: history_len[idx[low:low + curr_batch_size]],
                    self.response_ph: response[idx[low:low + curr_batch_size]],
                    # self.response_len_ph: response_len[idx[low:low + curr_batch_size]],
                    self.y_true: labels[idx[low: low + curr_batch_size]]
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch * len(idx) + low)
                low += curr_batch_size
                if low % (len(idx)//10) == 0:
                    print("epoch: {} iter: {} train_loss: {:g}".format(epoch, low, sess.run(self.total_loss, feed_dict=feed_dict)))
                if low % self.validation_step == 0:
                    print("Validation:")
                    if self.version == 1:
                        self.evaluate_from_n_utterances(sess)
                        print("Test:")
                        self.test(sess)
                    elif self.version == 2:
                        self.evaluate_from_n_utterances(sess)
                        print('Test:')
                        self.test(sess)
                    elif self.version == 3:
                        # only evaluate for v3 because we don't have test data yet!
                        self.evaluate_from_n_utterances(sess, n_utt=100, indices=indices_100)
                        self.evaluate_from_n_utterances(sess, n_utt=10, indices=indices_10)

                if low >= history.shape[0]:
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    epoch += 1

                    print('\nepoch={i}'.format(i=epoch))
                    print('reshuffle train data')
                    idx = np.array([i for i in range(len(history))])
                    np.random.shuffle(idx)

    def get_dims_and_recall(self, length, k=10):
        return length//k, k, [1, 2, 5] if k == 10 else [1, 2, 5, 10, 50]

    def evaluate_from_n_utterances(self, sess, n_utt=10, indices=None):
        self.all_candidate_scores = []
        history, history_len = self.valid_data_context, self.valid_data_context_len
        response, response_len = self.valid_data_response, self.valid_data_response_len
        low = 0

        if self.version in [1, 2]:
            indices = [i for i in range(len(history))]  # no need to shuffle indices r10@k for v1
        while True:
            feed_dict = {self.utterance_ph: history[indices[low:low + self.valid_batch_size]],
                         # self.all_utterance_len_ph: history_len[indices[low:low + self.valid_batch_size]],
                         self.response_ph: response[indices[low:low + self.valid_batch_size]],
                         # self.response_len_ph: response_len[indices[low:low + self.valid_batch_size]]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= len(indices):
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices))
        elif self.version == 3:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices), k=n_utt)

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def test(self, sess):
        self.all_candidate_scores = []
        history, history_len = self.test_data_context, self.test_data_context_len
        response, response_len = self.test_data_response, self.test_data_response_len
        low = 0
        while True:
            feed_dict = {self.utterance_ph: history[low:low + self.valid_batch_size],
                         # self.all_utterance_len_ph: history_len[low:low + self.valid_batch_size],
                         self.response_ph: response[low:low + self.valid_batch_size],
                         # self.response_len_ph: response_len[low:low + self.valid_batch_size]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(history))
        elif self.version == 3:
            pass  # no test data for DSTC7 yet!

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def evaluate_recall(self, y, k=1):
        num_examples = float(len(y))
        num_correct = 0

        for predictions in y:
            if 0 in predictions[:k]:
                num_correct += 1
        return num_correct / num_examples


class DeepAttentionMatchingNetwork():
    def __init__(self,
                 embedding_matrix=None,
                 train_data_context=None,
                 # train_data_context_turn_len=None,
                 train_data_context_len=None,
                 train_data_response=None,
                 train_data_response_len=None,
                 valid_data_context=None,
                 # valid_data_context_turn_len=None,
                 valid_data_context_len=None,
                 valid_data_response=None,
                 valid_data_response_len=None,
                 test_data_context=None,
                 # test_data_context_turn_len=None,
                 test_data_context_len=None,
                 test_data_response=None,
                 test_data_response_len=None,
                 train_labels=None,
                 num_words=200000,
                 batch_size=500,
                 valid_batch_size=200,
                 version=1,
                 trainable=False,
                 L=5,
                 is_positional=False
                 ):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        # self.total_words = 434511
        self.total_words = num_words + 1
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        self.version = version
        self.embedding_matrix = embedding_matrix

        self.train_data_context = train_data_context
        # self.train_data_context_turn_len = train_data_context_turn_len
        self.train_data_context_len = train_data_context_len
        self.train_data_response = train_data_response
        self.train_data_response_len = train_data_response_len

        self.valid_data_context = valid_data_context
        # self.valid_data_context_turn_len = valid_data_context_turn_len
        self.valid_data_context_len = valid_data_context_len
        self.valid_data_response = valid_data_response
        self.valid_data_response_len = valid_data_response_len

        # initialize test data if it exists
        self.test_data_context = test_data_context
        # self.test_data_context_turn_len = test_data_context_turn_len
        self.test_data_context_len = test_data_context_len
        self.test_data_response = test_data_response
        self.test_data_response_len = test_data_response_len

        self.train_labels = train_labels
        self.validation_step = len(self.train_data_context)

        self.trainable = trainable
        self.stack_num = L
        self.is_positional = is_positional

    def BuildModel(self):
        with tf.variable_scope('inputs'):
            # Utterances and their lengths
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
            # self.tt_turns_len = tf.placeholder(tf.int32, shape=(None, ))  # act. number of turns in context TODO: unused
            self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))

            # Responses and their lengths
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))

            # Labels
            self.y_true = tf.placeholder(tf.int32, shape=(None,))

            # Embeddings
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        with tf.variable_scope('embedding_matrix_init'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                          word_embedding_size), dtype=tf.float32, trainable=self.trainable)
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
        with tf.variable_scope('embedding_lookup'):
            all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
            response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        # with tf.variable_scope('context_unstack'):
        #     all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        # all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)  # list of 10 length

        Hr = response_embeddings
        if self.is_positional and self.stack_num > 0:
            with tf.variable_scope('positional'):
                Hr = op.positional_encoding_vector(Hr, max_timescale=10)

        Hr_stack = [Hr]

        for index in range(self.stack_num):
            with tf.variable_scope('self_stack_' + str(index)):
                Hr = layers.block(
                    Hr, Hr, Hr,
                    Q_lengths=self.response_len_ph, K_lengths=self.response_len_ph)
                Hr_stack.append(Hr)

        # context part
        # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
        list_turn_t = tf.unstack(self.utterance_ph, axis=1)
        list_turn_length = tf.unstack(self.all_utterance_len_ph, axis=1)

        sim_turns = []
        # for every turn_t calculate matching vector
        for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
            Hu = tf.nn.embedding_lookup(word_embeddings, turn_t)  # [batch, max_turn_len, emb_size]

            if self.is_positional and self.stack_num > 0:
                with tf.variable_scope('positional', reuse=True):
                    Hu = op.positional_encoding_vector(Hu, max_timescale=10)
            Hu_stack = [Hu]

            for index in range(self.stack_num):
                with tf.variable_scope('self_stack_' + str(index), reuse=True):
                    Hu = layers.block(
                        Hu, Hu, Hu,
                        Q_lengths=t_turn_length, K_lengths=t_turn_length)

                    Hu_stack.append(Hu)

            r_a_t_stack = []
            t_a_r_stack = []
            for index in range(self.stack_num + 1):

                with tf.variable_scope('t_attend_r_' + str(index)):
                    try:
                        t_a_r = layers.block(
                            Hu_stack[index], Hr_stack[index], Hr_stack[index],
                            Q_lengths=t_turn_length, K_lengths=self.response_len_ph)
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        t_a_r = layers.block(
                            Hu_stack[index], Hr_stack[index], Hr_stack[index],
                            Q_lengths=t_turn_length, K_lengths=self.response_len_ph)

                with tf.variable_scope('r_attend_t_' + str(index)):
                    try:
                        r_a_t = layers.block(
                            Hr_stack[index], Hu_stack[index], Hu_stack[index],
                            Q_lengths=self.response_len_ph, K_lengths=t_turn_length)
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        r_a_t = layers.block(
                            Hr_stack[index], Hu_stack[index], Hu_stack[index],
                            Q_lengths=self.response_len_ph, K_lengths=t_turn_length)

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            t_a_r = tf.stack(t_a_r_stack, axis=-1)
            r_a_t = tf.stack(r_a_t_stack, axis=-1)

            print(t_a_r, r_a_t)  # debug

            # calculate similarity matrix
            with tf.variable_scope('similarity'):
                # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                # divide sqrt(200) to prevent gradient explosion
                sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(200.0)

            sim_turns.append(sim)

        # cnn and aggregation
        sim = tf.stack(sim_turns, axis=1)
        print('sim shape: %s' % sim.shape)
        with tf.variable_scope('cnn_aggregation'):
            final_info = layers.CNN_3d(sim, 32, 16)
            # for douban
            # final_info = layers.CNN_3d(sim, 16, 16)

        variable_summaries(final_info, 'flatten/')

        # loss and train
        with tf.variable_scope('loss'):
            self.total_loss, self.logits = layers.loss(final_info, self.y_true, clip_value=1e15)
            self.y_pred = tf.nn.softmax(self.logits, name="y_pred")
            tf.summary.scalar('loss', self.total_loss)

            self.global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = 0.001
            # self.learning_rate = tf.constant(initial_learning_rate, dtype=tf.float32)
            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step=self.global_step,
                decay_steps=400,
                decay_rate=0.9,
                staircase=True)
            # values = [1e-3, 1e-4, 1e-5]
            # boundaries = [10000, 20000]
            # self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            # piecewise_constant works poorer

            Optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer = Optimizer.minimize(
            #     self.total_loss,
            #     global_step=self.global_step)

            # self.init = tf.global_variables_initializer()
            # self.saver = tf.train.Saver(max_to_keep=15)
            # self.all_variables = tf.global_variables()
            # self.all_operations = self._graph.get_operations()
            self.grads_and_vars = Optimizer.compute_gradients(self.total_loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    print(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1e15, 1e15), var) for grad, var in self.grads_and_vars]
            self.train_op = Optimizer.apply_gradients(
                self.capped_gvs,
                global_step=self.global_step)

        # Debug
        print_number_of_parameters()

    def restore_from_epoch(self, epoch_n):
        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess=sess, save_path="model/model.{}".format(epoch_n))
        return sess

    def create_indices(self, data_length, shuffle=True):
        indices_10 = []
        indices_100 = []
        idx = 0
        shift = 100
        epoch = 0
        indices = [i for i in range(1, 100)]
        while epoch < data_length // shift:
            shuffled_indices = [i for i in range(1, 100)]
            if shuffle:
                np.random.shuffle(shuffled_indices)
            indices_10.append(idx)
            indices_10.extend([i + shift * epoch for i in shuffled_indices[:9]])  # shuffle for r10@k
            indices_100.append(idx)
            indices_100.extend([i + shift * epoch for i in indices])  # do not shuffle for r100@k
            idx += shift
            epoch += 1

        return indices_10, indices_100

    def train(self, countinue_train=False, previous_modelpath="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('DAM_Graph', sess.graph)
            labels = self.train_labels
            history, history_len = self.train_data_context, self.train_data_context_len
            response, response_len = self.train_data_response, self.train_data_response_len

            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding_matrix})
            else:
                saver.restore(sess, previous_modelpath)
            low = 0
            epoch = 1
            indices_10, indices_100 = self.create_indices(data_length=len(self.valid_data_context),
                                                          shuffle=True
                                                          )  # valid indices
            print('\nshuffle train data, len: {}'.format(len(history)))
            idx = np.array([i for i in range(len(history))])
            np.random.shuffle(idx)
            print('epoch', epoch)
            while epoch <= 15:
                curr_batch_size = min(low + self.batch_size, history.shape[0]) - low
                feed_dict = {
                    self.utterance_ph: history[idx[low:low + curr_batch_size]],
                    self.all_utterance_len_ph: history_len[idx[low:low + curr_batch_size]],
                    self.response_ph: response[idx[low:low + curr_batch_size]],
                    self.response_len_ph: response_len[idx[low:low + curr_batch_size]],
                    self.y_true: labels[idx[low: low + curr_batch_size]]
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch * len(idx) + low)
                low += curr_batch_size
                if low % (len(idx)//10) == 0:
                    l, lr = sess.run([self.total_loss, self.learning_rate], feed_dict=feed_dict)
                    print("epoch: {} iter: {} train_loss: {:g}, lr: {:g}".format(epoch, low, l, lr))
                if low % self.validation_step == 0:
                    print("Validation:")
                    if self.version == 1:
                        self.evaluate_from_n_utterances(sess)
                        print("Test:")
                        self.test(sess)
                    elif self.version == 2:
                        self.evaluate_from_n_utterances(sess)
                        print('Test:')
                        self.test(sess)
                    elif self.version == 3:
                        # only evaluate for v3 because we don't have test data yet!
                        self.evaluate_from_n_utterances(sess, n_utt=100, indices=indices_100)
                        self.evaluate_from_n_utterances(sess, n_utt=10, indices=indices_10)

                if low >= history.shape[0]:
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    epoch += 1

                    print('\nepoch={i}'.format(i=epoch))
                    print('reshuffle train data')
                    idx = np.array([i for i in range(len(history))])
                    np.random.shuffle(idx)

    def get_dims_and_recall(self, length, k=10):
        return length//k, k, [1, 2, 5] if k == 10 else [1, 2, 5, 10, 50]

    def evaluate_from_n_utterances(self, sess, n_utt=10, indices=None):
        self.all_candidate_scores = []
        history, history_len = self.valid_data_context, self.valid_data_context_len
        response, response_len = self.valid_data_response, self.valid_data_response_len
        low = 0

        if self.version in [1, 2]:
            indices = [i for i in range(len(history))]  # no need to shuffle indices r10@k for v1
        while True:
            feed_dict = {self.utterance_ph: history[indices[low:low + self.valid_batch_size]],
                         self.all_utterance_len_ph: history_len[indices[low:low + self.valid_batch_size]],
                         self.response_ph: response[indices[low:low + self.valid_batch_size]],
                         self.response_len_ph: response_len[indices[low:low + self.valid_batch_size]]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= len(indices):
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices))
        elif self.version == 3:
            dim1, dim2, recalls = self.get_dims_and_recall(len(indices), k=n_utt)

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def test(self, sess):
        self.all_candidate_scores = []
        history, history_len = self.test_data_context, self.test_data_context_len
        response, response_len = self.test_data_response, self.test_data_response_len
        low = 0
        while True:
            feed_dict = {self.utterance_ph: history[low:low + self.valid_batch_size],
                         self.all_utterance_len_ph: history_len[low:low + self.valid_batch_size],
                         self.response_ph: response[low:low + self.valid_batch_size],
                         self.response_len_ph: response_len[low:low + self.valid_batch_size]
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + self.valid_batch_size
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        dim1 = dim2 = recalls = None
        if self.version in [1, 2]:
            dim1, dim2, recalls = self.get_dims_and_recall(len(history))
        elif self.version == 3:
            pass  # no test data for DSTC7 yet!

        y = np.array(all_candidate_scores).reshape(dim1, dim2)
        y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
        for n in recalls:
            print('Recall @ ({}, {}): {:g}'.format(n, dim2, self.evaluate_recall(y, n)))

    def evaluate_recall(self, y, k=1):
        num_examples = float(len(y))
        num_correct = 0

        for predictions in y:
            if 0 in predictions[:k]:
                num_correct += 1
        return num_correct / num_examples

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=200000, type=int)
    psr.add_argument('--word_dim', default=200, type=int)
    psr.add_argument('--sent_dim', default=200, type=int)
    psr.add_argument('--last_dim', default=50, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--train_data_context', default='train_context.joblib')
    psr.add_argument('--train_data_response', default='train_response.joblib')
    psr.add_argument('--valid_data_context', default='valid_context.joblib')
    psr.add_argument('--valid_data_response', default='valid_response.joblib')
    psr.add_argument('--test_data_context', default='test_context.joblib')
    psr.add_argument('--test_data_response', default='test_response.joblib')
    psr.add_argument('--model_name', default='SMN_last')
    psr.add_argument('--batch_size', default=500, type=int)
    psr.add_argument('--valid_batch_size', default=200, type=int)
    psr.add_argument('--version', default=1, type=int)
    psr.add_argument('--elmo', default='off')
    psr.add_argument('--trainable', default='no')
    psr.add_argument('--dam', default='no')
    psr.add_argument('--dam_old', default='no')
    psr.add_argument('--restore', default='no')
    psr.add_argument('--L', type=int, default=5, help='stack number')
    psr.add_argument('--is_positional', default='no')
    args = psr.parse_args()

    if args.dam in ['yes', 'on']:
        print('load embedding matrix')
        embedding_matrix = joblib.load(args.embedding_matrix)

        print('load train data')
        print('load context')
        train_data = joblib.load(args.train_data_context)
        context = np.array(train_data['context'])
        context_len = np.array(train_data['context_len'])
        print('load response')
        train_data = joblib.load(args.train_data_response)
        response = np.array(train_data['response'])
        response_len = np.array(train_data['response_len'])
        labels = train_data['labels']

        print('load valid data')
        print('load context')
        valid_data = joblib.load(args.valid_data_context)
        valid_context = np.array(valid_data['context'])
        valid_context_len = np.array(valid_data['context_len'])
        print('load response')
        valid_data = joblib.load(args.valid_data_response)
        valid_response = np.array(valid_data['response'])
        valid_response_len = np.array(valid_data['response_len'])

        print('load test data')
        print('load context')
        test_data = joblib.load(args.test_data_context)
        test_context = np.array(test_data['context'])
        test_context_len = np.array(test_data['context_len'])
        print('load response')
        test_data = joblib.load(args.test_data_response)
        test_response = np.array(test_data['response'])
        test_response_len = np.array(test_data['response_len'])

        # # calculate actual lengths of contexts:
        # context_turns_len = np.zeros(shape=(len(context_len), ))
        # for index, sentences_len in context_len:
        #     context_turns_len[index] = len(sentences_len[sentences_len != 0])
        #
        # valid_context_turns_len = np.zeros(shape=(len(valid_context_len),))
        # for index, sentences_len in valid_context_len:
        #     valid_context_turns_len[index] = len(sentences_len[sentences_len != 0])
        #
        # test_context_turns_len = np.zeros(shape=(len(test_context_len),))
        # for index, sentences_len in test_context_len:
        #     test_context_turns_len[index] = len(sentences_len[sentences_len != 0])

        # context_len = None
        # context = None
        # response = None
        # response_len = None
        # embedding_matrix = None
        # valid_context = None
        # valid_context_len = None
        # valid_response = None
        # valid_response_len = None
        # test_context = None
        # test_context_len = None
        # test_response = None
        # test_response_len = None
        # test_context = None
        # labels = None

        model = DeepAttentionMatchingNetwork(num_words=args.num_words,
                                             embedding_matrix=embedding_matrix,

                                             train_data_context=context,
                                             # train_data_context_turn_len=context_turns_len,
                                             train_data_context_len=context_len,
                                             train_data_response=response,
                                             train_data_response_len=response_len,

                                             valid_data_context=valid_context,
                                             # valid_data_context_turn_len=valid_context_turns_len,
                                             valid_data_context_len=valid_context_len,
                                             valid_data_response=valid_response,
                                             valid_data_response_len=valid_response_len,

                                             test_data_context=test_context,
                                             # test_data_context_turn_len=test_context_turns_len,
                                             test_data_context_len=test_context_len,
                                             test_data_response=test_response,
                                             test_data_response_len=test_response_len,

                                             train_labels=labels,
                                             batch_size=args.batch_size,
                                             version=args.version,

                                             trainable=(True if args.trainable is 'yes' else False),
                                             L=args.L,
                                             is_positional=(True if args.is_positional in ['yes', 'on'] else False)
                                             )

        print('building model')
        model.BuildModel()

        if args.restore in ['yes', 'on']:
            print('restoring')
            model.train(countinue_train=True, previous_modelpath="model/model.15")
        else:
            print('fitting')
            model.train()
        exit()

    if args.dam_old in ['yes', 'on']:
        print('load embedding matrix')
        embedding_matrix = joblib.load(args.embedding_matrix)

        print('load train data')
        print('load context')
        train_data = joblib.load(args.train_data_context)
        context = np.array(train_data['context'])
        context_len = np.array(train_data['context_len'])
        print('load response')
        train_data = joblib.load(args.train_data_response)
        response = np.array(train_data['response'])
        response_len = np.array(train_data['response_len'])
        labels = train_data['labels']

        print('load valid data')
        print('load context')
        valid_data = joblib.load(args.valid_data_context)
        valid_context = np.array(valid_data['context'])
        valid_context_len = np.array(valid_data['context_len'])
        print('load response')
        valid_data = joblib.load(args.valid_data_response)
        valid_response = np.array(valid_data['response'])
        valid_response_len = np.array(valid_data['response_len'])

        print('load test data')
        print('load context')
        test_data = joblib.load(args.test_data_context)
        test_context = np.array(test_data['context'])
        test_context_len = np.array(test_data['context_len'])
        print('load response')
        test_data = joblib.load(args.test_data_response)
        test_response = np.array(test_data['response'])
        test_response_len = np.array(test_data['response_len'])

        model = DeepAttentionMatchingNetwork_old(num_words=args.num_words,
                                             embedding_matrix=embedding_matrix,

                                             train_data_context=context,
                                             train_data_context_len=context_len,
                                             train_data_response=response,
                                             train_data_response_len=response_len,

                                             valid_data_context=valid_context,
                                             valid_data_context_len=valid_context_len,
                                             valid_data_response=valid_response,
                                             valid_data_response_len=valid_response_len,

                                             test_data_context=test_context,
                                             test_data_context_len=test_context_len,
                                             test_data_response=test_response,
                                             test_data_response_len=test_response_len,

                                             train_labels=labels,
                                             batch_size=args.batch_size,
                                             version=args.version,

                                             trainable=(True if args.trainable is 'yes' else False)
                                             )

        print('building model')
        model.BuildModel()

        if args.restore in ['yes', 'on']:
            print('restoring')
            model.train(countinue_train=True, previous_modelpath="model/model.15")
        else:
            print('fitting')
            model.train()
        exit()

    if args.elmo == 'off':
        print('load embedding matrix')
        embedding_matrix = joblib.load(args.embedding_matrix)

        print('load train data')
        print('load context')
        train_data = joblib.load(args.train_data_context)
        context = np.array(train_data['context'])
        context_len = np.array(train_data['context_len'])
        print('load response')
        train_data = joblib.load(args.train_data_response)
        response = np.array(train_data['response'])
        response_len = np.array(train_data['response_len'])
        labels = train_data['labels']

        print('load valid data')
        print('load context')
        valid_data = joblib.load(args.valid_data_context)
        valid_context = np.array(valid_data['context'])
        valid_context_len = np.array(valid_data['context_len'])
        print('load response')
        valid_data = joblib.load(args.valid_data_response)
        valid_response = np.array(valid_data['response'])
        valid_response_len = np.array(valid_data['response_len'])

        print('load test data')
        print('load context')
        test_data = joblib.load(args.test_data_context)
        test_context = np.array(test_data['context'])
        test_context_len = np.array(test_data['context_len'])
        print('load response')
        test_data = joblib.load(args.test_data_response)
        test_response = np.array(test_data['response'])
        test_response_len = np.array(test_data['response_len'])

        model = SCN(num_words=args.num_words,
                    embedding_matrix=embedding_matrix,

                    train_data_context=context,
                    train_data_context_len=context_len,
                    train_data_response=response,
                    train_data_response_len=response_len,

                    valid_data_context=valid_context,
                    valid_data_context_len=valid_context_len,
                    valid_data_response=valid_response,
                    valid_data_response_len=valid_response_len,

                    test_data_context=test_context,
                    test_data_context_len=test_context_len,
                    test_data_response=test_response,
                    test_data_response_len=test_response_len,

                    train_labels=labels,
                    batch_size=args.batch_size,
                    version=args.version,

                    trainable=(True if args.trainable is 'yes' else False)
                    )
        print('building model')
        model.BuildModel()

        print('fitting')
        model.train()

    elif args.elmo == 'on':
        # Don't use args paths ->

        print('load train data')
        print('load context/response/labels')
        train_data = joblib.load('train_context.joblib')
        context_len = np.array(train_data['context_len'])

        train_data = joblib.load('word_train_context.joblib')
        # context = np.empty((1, 10, 50))
        # for c in train_data['context']:
        #     context.append(np.array(c))
        # context = np.delete(context, (0), axis=0)
        context = train_data['context']
        response = train_data['response']
        # context = np.load('words_tr_c' + '.npz')
        # response = np.load('words_tr_r' + '.npz')

        train_data = joblib.load('train_response.joblib')
        response_len = np.array(train_data['response_len'])
        labels = train_data['labels']

        #######################################
        print('load valid data')
        print('load context/response/len')
        valid_data = joblib.load('valid_context.joblib')
        valid_context_len = np.array(valid_data['context_len'])

        valid_data = joblib.load('word_valid_context.joblib')
        valid_context = valid_data['context']
        valid_response = valid_data['response']
        # valid_context = np.load('words_va_c' + '.npz')
        # valid_response = np.load('words_va_r' + '.npz')

        print('load response')
        valid_data = joblib.load('valid_response.joblib')
        valid_response_len = np.array(valid_data['response_len'])

        ######################################
        print('load test data')
        print('load context/response/len')
        test_data = joblib.load('test_context.joblib')
        test_context_len = np.array(test_data['context_len'])

        test_data = joblib.load('word_test_context.joblib')
        test_context = test_data['context']
        test_response = test_data['response']
        # test_context = np.load('words_te_c' + '.npz')
        # test_response = np.load('words_te_r' + '.npz')

        print('load response')
        test_data = joblib.load('test_response.joblib')
        test_response_len = np.array(test_data['response_len'])


        model = SCN_ELMo(
            train_data_context=context,
            train_data_context_len=context_len,
            train_data_response=response,
            train_data_response_len=response_len,

            valid_data_context=valid_context,
            valid_data_context_len=valid_context_len,
            valid_data_response=valid_response,
            valid_data_response_len=valid_response_len,

            test_data_context=test_context,
            test_data_context_len=test_context_len,
            test_data_response=test_response,
            test_data_response_len=test_response_len,

            train_labels=labels,
            batch_size=args.batch_size,
            valid_batch_size=args.valid_batch_size,
            version=args.version,

            trainable=(True if args.trainable is 'yes' else False)
        )
        print('building model')
        model.BuildModel()

        print('fitting')
        model.train()

if __name__ == '__main__': main()
