import argparse
import numpy as np
np.random.seed(42)

import joblib
import tensorflow as tf
tf.set_random_seed(42)

from train_tf import SCN

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=200000, type=int)
    psr.add_argument('--word_dim', default=200, type=int)
    psr.add_argument('--sent_dim', default=200, type=int)
    psr.add_argument('--last_dim', default=50, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--valid_data_context', default='valid_context.joblib')
    psr.add_argument('--valid_data_response', default='valid_response.joblib')
    psr.add_argument('--test_data_context', default='test_context.joblib')
    psr.add_argument('--test_data_response', default='test_response.joblib')
    psr.add_argument('--batch_size', default=500, type=int)
    psr.add_argument('--version', default=1, type=int)
    psr.add_argument('--epoch_n', default=1, type=int)
    args = psr.parse_args()

    print('load embedding matrix')
    embedding_matrix = joblib.load(args.embedding_matrix)

    print('load valid data')
    print('load context')
    valid_data = joblib.load(args.valid_data_context)
    valid_context = np.array(valid_data['context'])
    valid_context_len = np.array(valid_data['context_len'])
    print('load response')
    valid_data = joblib.load(args.valid_data_response)
    valid_response = np.array(valid_data['response'])
    valid_response_len = np.array(valid_data['response_len'])

    model = None
    if args.version == 1:
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

                    valid_data_context=valid_context,
                    valid_data_context_len=valid_context_len,
                    valid_data_response=valid_response,
                    valid_data_response_len=valid_response_len,

                    test_data_context=test_context,
                    test_data_context_len=test_context_len,
                    test_data_response=test_response,
                    test_data_response_len=test_response_len,

                    version=args.version,
                    )
    elif args.version == 3:
        model = SCN(num_words=args.num_words,
                    embedding_matrix=embedding_matrix,

                    valid_data_context=valid_context,
                    valid_data_context_len=valid_context_len,
                    valid_data_response=valid_response,
                    valid_data_response_len=valid_response_len,

                    version=args.version,
                    )

    print('building model')
    model.BuildModel()

    print('restore session from epoch {}'.format(args.epoch_n))
    session = model.restore_from_epoch(args.epoch_n)

    print('evaluate the model for v{} valid data'.format(args.epoch_n))
    if args.version == 1:
        print('Validation:')
        model.evaluate_from_n_utterances(sess=session)
        print('Test:')
        model.test(sess=session)
    if args.version == 2:
        # TODO: evaluate for Ubuntu v2
        pass
    elif args.version == 3:
        print('Validation:')
        indices_10, indices_100 = model.create_indices(len(valid_context))
        model.evaluate_from_n_utterances(sess=session, n_utt=100, indices=indices_100)
        model.evaluate_from_n_utterances(sess=session, n_utt=10, indices=indices_10)


if __name__ == '__main__': main()
