import argparse
import numpy as np

np.random.seed(42)

import joblib
import tensorflow as tf

tf.set_random_seed(42)


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
        self.validation_step = len(self.train_data_context) // 2

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
            print('\nshuffle train data')
            idx = np.array([i for i in range(len(history))])
            np.random.shuffle(idx)
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
                train_writer.add_summary(summary, (epoch - 1) * history.shape[0] // self.batch_size + low)
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
    psr.add_argument('--version', default=1, type=int)
    args = psr.parse_args()

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

                trainable=False
                )
    print('building model')
    model.BuildModel()

    print('fitting')
    model.train()


if __name__ == '__main__': main()
