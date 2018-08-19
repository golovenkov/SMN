import argparse
import numpy as np
np.random.seed(42)

import joblib
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
tf.set_random_seed(42)
from keras import backend as K
from model import build_SMN


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
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--word_dim', default=200, type=int)
    psr.add_argument('--sent_dim', default=200, type=int)
    psr.add_argument('--last_dim', default=50, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--train_data', default='train.joblib')
    psr.add_argument('--valid_data', default='valid.joblib')
    psr.add_argument('--model_name', default='SMN_last')
    psr.add_argument('--batch_size', default=512, type=int)
    args = psr.parse_args()

    print('load embedding matrix')
    embedding_matrix = joblib.load(args.embedding_matrix)

    print('build model')
    model = build_SMN(args.max_turn, args.maxlen, args.word_dim, args.sent_dim, args.last_dim, args.num_words, embedding_matrix)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # json_string = model.to_json()
    # open(args.model_name + '.json', 'w').write(json_string)

    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model_checkpoint = ModelCheckpoint(args.model_name + '.h5', save_best_only=True, save_weights_only=True)

    print('load train data')
    train_data = joblib.load(args.train_data)
    context = np.array(train_data['context'])
    response = np.array(train_data['response'])
    labels = train_data['labels']

    print('load valid data')
    valid_data = joblib.load(args.valid_data)
    valid_context = np.array(valid_data['context'])
    valid_response = np.array(valid_data['response'])
    valid_labels = valid_data['labels']

    print('fitting')
    model.fit(
        [context, response],
        labels,
        validation_data=([valid_context, valid_response], valid_labels),
        batch_size=args.batch_size,
        epochs=10,
        callbacks=[early_stopping, model_checkpoint]
    )

if __name__ == '__main__': main()
