import numpy as np
import joblib
import argparse
import codecs
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec, Text8Corpus
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import ijson
import numpy as np

def create_dialog_iter(filename, mode="train", sampling="10-10"):
    """
    Returns an iterator over a JSON file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:

            dialog = entry
            rows = []
            message = ''
            utterances = dialog['messages-so-far']
            for msg in utterances:
                message += msg['utterance']
                message += ' _eot_ '

            # true response
            true_response = dialog['options-for-correct-answers'][0]['utterance']

            fake_responses = []
            correct_answer = dialog['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
            for i, utterance in enumerate(dialog['options-for-next']):
                if utterance['candidate-id'] != target_id:
                    fake_responses.append(utterance['utterance'])

            true = (message, true_response, 1)
            if mode == 'train':
                if sampling == '1-9':
                    rows.append(true)
                    for fake_response in np.random.choice(fake_responses, 9):
                        rows.append((message, fake_response, 0))
                elif sampling == '10-10':
                    for fake_response in np.random.choice(fake_responses, 10):
                        rows.append(true)
                        rows.append((message, fake_response, 0))
                elif sampling == '20-20':
                    for fake_response in np.random.choice(fake_responses, 20):
                        rows.append(true)
                        rows.append((message, fake_response, 0))
                elif sampling == '30-30':
                    for fake_response in np.random.choice(fake_responses, 30):
                        rows.append(true)
                        rows.append((message, fake_response, 0))

            elif mode == 'valid':
                # return a list of 100 tuples
                rows.append(true)
                for fake_response in fake_responses:
                    rows.append((message, fake_response, 0))
                # rows.append(true)
                # for fake_response in np.random.choice(fake_responses, 9):
                #     rows.append((message, fake_response, 0))

            # need to return [(message, response, label), ...]
            # print(len(rows))
            yield rows

def build_multiturn_data(multiturn_data, version=1, mode="train", sampling='10-10'):
    contexts = []
    responses = []
    labels = []
    if version == 1:
        with codecs.open(multiturn_data,'r','utf-8') as f:
            for line in tqdm(f):
                line = line.replace('_','')
                parts = line.strip().split('\t')

                lable = parts[0]
                message = ''
                for i in range(1, len(parts)-1, 1):
                    message += parts[i]
                    message += ' _eot_ '

                response = parts[-1]

                contexts.append(message)
                responses.append(response)
                labels.append(lable)
    elif version == 2:
        for samples in create_dialog_iter(multiturn_data, mode, sampling):
            for sample in samples:
                contexts.append(sample[0])
                responses.append((sample[1]))
                labels.append(sample[2])

    return contexts, responses, np.array(labels)

def preprocess_texts(texts, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)

def preprocess_multi_turn_texts(context, max_turn, maxlen):
    multi_turn_texts = []
    for i in tqdm(range(len(context))):
        multi_turn_texts.append(context[i].split('_eot_')[-(max_turn+1):-1])
        if len(multi_turn_texts[i]) <= max_turn:
            tmp = multi_turn_texts[i][:]
            multi_turn_texts[i] = [' '] * (max_turn - len(multi_turn_texts[i]))
            multi_turn_texts[i].extend(tmp)

    multi_turn_texts = [preprocess_texts(multi_turn_texts[i], maxlen) for i in tqdm(range(len(multi_turn_texts)))]
    return multi_turn_texts

def word2vec_embedding(path, num_words, embedding_dim, word_index):
    w2v = Word2Vec.load(path)
    # w2v = Word2Vec.load_word2vec_format(path)
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(path)
    num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = w2v[word]
        except KeyError:
            pass
    return embedding_matrix

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--embedding_dim', default=200, type=int)
    psr.add_argument('--w2v_path', default='ubuntu_word2vec_200.model')
    psr.add_argument('--train_data', default='ubuntu_data/train.txt')
    psr.add_argument('--valid_data', default='ubuntu_data/valid.txt')
    psr.add_argument('--test_data', default='ubuntu_data/test.txt')
    psr.add_argument('--version', default=1, type=int)
    psr.add_argument('--sampling', default='10-10', type=str)
    args = psr.parse_args()

    print('load data')
    if args.version == '1':
        train_context, train_response, train_labels = build_multiturn_data(args.train_data)
        valid_context, valid_response, valid_labels = build_multiturn_data(args.valid_data)
        test_context, test_response, test_labels = build_multiturn_data(args.test_data)
    else:
        train_context, train_response, train_labels = build_multiturn_data(args.train_data, args.version, "train", args.sampling)
        valid_context, valid_response, valid_labels = build_multiturn_data(args.valid_data, args.version, "valid", args.sampling)


    print('tokenize')
    global tokenizer, maxlen
    tokenizer = Tokenizer(num_words=args.num_words, filters="\t\n,", split=' ')
    # tokenizer.fit_on_texts(np.append(train_context, train_response))  # numpy can throw MemoryError here
    if args.version == 1:
        tokenizer.fit_on_texts(np.append(train_context, train_response))
    elif args.version == 2:
        sentences = Text8Corpus("ubuntu_data_v2/train.txt")
        tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    print('create word matrix')
    embedding_matrix = word2vec_embedding(path=args.w2v_path, num_words=args.num_words, embedding_dim=args.embedding_dim, word_index=word_index)

    print('preprocess data')
    train_context = preprocess_multi_turn_texts(train_context, args.max_turn, args.maxlen)
    train_response = preprocess_texts(train_response, args.maxlen)
    valid_context = preprocess_multi_turn_texts(valid_context, args.max_turn, args.maxlen)
    valid_response = preprocess_texts(valid_response, args.maxlen)
    if args.version == 1:
        test_context = preprocess_multi_turn_texts(test_context, args.max_turn, args.maxlen)
        test_response = preprocess_texts(test_response, args.maxlen)

    train_data = {'context': train_context, 'response': train_response, 'labels': train_labels}
    valid_data = {'context': valid_context, 'response': valid_response, 'labels': valid_labels}
    if args.version == 1:
        test_data = {'context': test_context, 'response': test_response, 'labels': test_labels}

    print('dump')
    joblib.dump(train_data, 'train.joblib', protocol=-1, compress=3)
    joblib.dump(valid_data, 'valid.joblib', protocol=-1, compress=3)
    if args.version == 1:
        joblib.dump(test_data, 'test.joblib', protocol=-1, compress=3)
    joblib.dump(embedding_matrix, 'embedding_matrix.joblib', protocol=-1, compress=3)

if __name__ == '__main__': main()
