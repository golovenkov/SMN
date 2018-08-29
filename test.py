import argparse
import joblib
import numpy as np
np.random.seed(42)

from keras.models import model_from_json
from model import *
import tensorflow as tf
tf.set_random_seed(42)
from keras import backend as K

def evaluate_recall(y, k=1):
    num_examples = float(len(y))
    num_correct = 0

    for predictions in y:
        if 0 in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

def main():
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))

    psr = argparse.ArgumentParser()
    psr.add_argument('--test_data', default='test.joblib')
    psr.add_argument('--model_name', default='SMN_last')
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--version', default=1, type=int)
    psr.add_argument('--batch_size', default=2000, type=int)
    args = psr.parse_args()

    print('load data')
    test_data = joblib.load(args.test_data)

    print('load embedding matrix')
    embedding_matrix = joblib.load(args.embedding_matrix)

    # json_string = open(args.model_name + '.json').read()
    # model = model_from_json(json_string)

    print('build model')
    model = build_SMN(10, 50, 200, 200, 50, args.num_words, embedding_matrix)
    model.load_weights(args.model_name + '.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    context = np.array(test_data['context'])
    response = np.array(test_data['response'])

    print(context[0], response[0])

    print('predict')
    y = model.predict([context, response], batch_size=args.batch_size, verbose=1)

    if args.version == 1:
        dim1 = 50000
        dim2 = 10
        recalls = [1, 2, 5]
    elif args.version == 2:
        dim1 = 5000
        dim2 = 100
        recalls = [1, 2, 5, 10, 50, 100]
        # below is for grouping in 10 samples
        # dim1 = 5000
        # dim2 = 10
        # recalls = [1, 2, 5]
    y = np.array(y).reshape(dim1, dim2)
    y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
    for n in recalls:
        print('Recall @ ({}, {}): {:g}'.format(n, dim2, evaluate_recall(y, n)))
    
if __name__ == '__main__': main()
