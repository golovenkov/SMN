import joblib
import argparse
import codecs
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec, Text8Corpus
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import ijson
import numpy as np
import pickle
from multiprocessing import Pool
import fastText as Fasttext

NEED_WORDS = False

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
                # message += " ".join(process_line(msg['utterance']))  # process_line
                message += msg['utterance']
                message += ' _eot_ '

            # true response
            # true_response = " ".join(process_line(dialog['options-for-correct-answers'][0]['utterance']))  # process line
            true_response = dialog['options-for-correct-answers'][0]['utterance']

            fake_responses = []
            correct_answer = dialog['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
            for i, utterance in enumerate(dialog['options-for-next']):
                if utterance['candidate-id'] != target_id:
                    # fake_responses.append(" ".join(process_line(utterance['utterance'])))  # preprocess line
                    fake_responses.append(utterance['utterance'])

            true = (message, true_response, 1)
            if mode == 'train':
                if sampling == '1-1':
                    rows.append(true)
                    rows.append((message, np.random.choice(fake_responses), 0))  # random 1 from 99
                elif sampling == '1-9':
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
                for r in fake_responses:
                    rows.append((message, r, 0))
                # rows.append(true)
                # for fake_response in np.random.choice(fake_responses, 9):
                #     rows.append((message, fake_response, 0))

            # need to return [(message, response, label), ...]
            # print(len(rows))
            yield rows

def map_process_line(x):
    return remove_punctuation(" ".join(process_line(x, clean_string=True)))

import pandas as pd

def build_multiturn_data(multiturn_data, version=1, mode="train", sampling='10-10'):
    """
    Parse the source dataset files and split into context/response/train

    multiturn_data - text file to process (for example, train.txt)
    version        - if the source file is Ubuntu Dialogue Corpus v1 / v2 / v3
    mode           - if the source file for train
    sampling       - the type of example sampling for training if more than one a negative sample is provided
    """
    contexts = []
    responses = []
    labels = []
    if version == 1:
        with codecs.open(multiturn_data,'r','utf-8') as f:
            for line in tqdm(f):
                line = line.replace('_','')
                parts = line.strip().split('\t')

                label = int(parts[0])
                message = ''
                for i in range(1, len(parts)-1, 1):
                    message += parts[i]
                    message += ' _eot_ '

                response = parts[-1]

                contexts.append(message)
                responses.append(response)
                labels.append(label)
    elif version == 2:
        # TODO: parse csv files for Ubuntu v2
        df = pd.read_csv(multiturn_data)
        if len(df.columns) == 3:  # train
            for index, row in tqdm(df.iterrows()):
                # if index > 1000: break                  # TODO: remove

                # merge sentences in each turn
                message = row.Context.replace("__eou__", "")
                response = row.Utterance.replace("__eou__", "")
                label = int(float(row.Label))

                contexts.append(message)
                responses.append(response)
                labels.append(label)
        elif len(df.columns) == 11:
            for index, row in tqdm(df.iterrows()):
                # if index > 1000: break  # TODO: remove

                # merge sentences in each turn
                message = row[0].replace("__eou__", "")

                for i in range(1, 11):
                    response = row[i].replace("__eou__", "")
                    label = 1 if i == 1 else 0

                    contexts.append(message)
                    responses.append(response)
                    labels.append(label)
        pool = Pool()
        prep_contexts = [*pool.map(map_process_line, tqdm(contexts), chunksize=1000)]
        prep_responses = [*pool.map(map_process_line, tqdm(responses), chunksize=1000)]
        pool.terminate()

        contexts = prep_contexts
        responses = prep_responses
    elif version == 3:
        # i = 0
        for samples in tqdm(create_dialog_iter(multiturn_data, mode, sampling)):
            # if i > 100: break
            for sample in samples:
                contexts.append(sample[0])
                responses.append((sample[1]))
                labels.append(sample[2])
            # i += 1

        pool = Pool()
        prep_contexts = [*pool.map(map_process_line, tqdm(contexts), chunksize=1000)]
        prep_responses = [*pool.map(map_process_line, tqdm(responses), chunksize=1000)]
        pool.terminate()

        contexts = prep_contexts
        responses = prep_responses

    return contexts, responses, np.array(labels)


def word2id(c):
    if c in word_index:
        return word_index[c]
    else:
        return word_index['<UNK>']   # return last index, usually <UNK> or mean of all words

# def tokenize_utterance(utt):
#     return list(map(word2id, text_to_word_sequence(remove_stop_words(utt), split=" ")))

def pad_words(X, maxlen):
    new_seq = []
    for i in range(maxlen):
        try:
            new_seq.append(X[i])
        except:
            new_seq.append("")   # PAD with an empty string
    return new_seq

def preprocess_texts(texts):
    """ Tokenize and zero-pad the sentence.

    For tokenizing is used the 'tokenizer' class instance.
    For padding is used 'post' padding

    text - list of sentences
    maxlen - maximum length of sentence
    word_index - mapping 'word to int'

    Returns:
        a tuple (sequences, sequences_length), where
        * sequences - 2-D tokenized list of sentences
        * sequences_length - 1-D array of lengths of the sentences

    Note: if the length of the sequence is less than maxlen setting the length equal to maxlen
    """
    maxlen = 50
    sequences = [list(map(word2id, text_to_word_sequence(each_utt, filters=filters, split=" "))) for each_utt in texts]
    if NEED_WORDS:
        word_sequences = [pad_words(text_to_word_sequence(each_utt, filters=filters, split=" "), maxlen) for each_utt in texts]
    else:
        word_sequences = None

    # sequences = []
    # for each_utt in texts:
    #     sequence = np.array(list(map(word2id, text_to_word_sequence(each_utt, filters=filters, split=" "))))
    #     sequence = sequence[sequence != 0]   # filter zeros
    #     sequences.append(sequence)
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return pad_sequences(sequences, maxlen=maxlen, padding="post"), sequences_length, word_sequences


def preprocess_multi_turn_texts(context, max_turn):
    print('Trim or pad to max_turn utterances')
    multi_turn_texts = []
    for i in tqdm(range(len(context))):
        multi_turn_texts.append(context[i].split('_eot_')[-(max_turn+1):-1])
        if len(multi_turn_texts[i]) <= max_turn:
            tmp = multi_turn_texts[i][:]
            multi_turn_texts[i] = [' '] * (max_turn - len(multi_turn_texts[i]))
            multi_turn_texts[i].extend(tmp)

    print('Tokenize and pad the sentences')
    # tokenized_multi_turn_texts = []
    # sequences_length = []
    # all_padded_word_sequences = []
    # for i in tqdm(range(len(multi_turn_texts))):
    #     tokenized_sentences, tokenized_sentences_lengths, padded_word_sequences = preprocess_texts(multi_turn_texts[i])
    #     tokenized_multi_turn_texts.append(tokenized_sentences)
    #     sequences_length.append(tokenized_sentences_lengths)
    #     all_padded_word_sequences.append(padded_word_sequences)

    pool = Pool()
    allinone_tuple = [*pool.map(preprocess_texts, tqdm(multi_turn_texts), chunksize=1000)]
    pool.terminate()
    tokenized_multi_turn_texts = []
    sequences_length = []
    all_padded_word_sequences = []
    for each_tuple in tqdm(allinone_tuple):
        tokenized_multi_turn_texts.append(each_tuple[0])
        sequences_length.append(each_tuple[1])
        all_padded_word_sequences.append(each_tuple[2])

    return tokenized_multi_turn_texts, sequences_length, all_padded_word_sequences

def word2vec_embedding(path, num_words, embedding_dim, word_index):
    """
    Create embedding matrix from the given word2vec embeddings using maximum 'num_words'

    path - path to the file
    num_words - maximum number of words from word_index to use
    embedding_dim - dim of embeddings, usually 200
    word-index - hash-map for mapping words to indices
    """
    model = Word2Vec.load(path)
    # w2v = Word2Vec.load_word2vec_format(path)
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(path)
    num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    not_found_words = []
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = model[word]   # try to find a vector for the given word
        except KeyError:
            not_found_words.append(word)
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:num_words], axis=0)               # <UNK> token as mean of all the vectors
    if len(not_found_words) > 0:
        print('Not found embedding vectors for {} words out of {}'.format(len(not_found_words), len(word_index)))
        print(" ".join(not_found_words))
    return embedding_matrix

def fasttext_embeddings(path, num_words, embedding_dim, word_index):
    """
    Create embeddings matrix from the given fastText embeddings using maximum 'num_words'
    :param path: path to the file
    :param num_words: maximum number of words from word_index to use
    :param embedding_dim: dim of embeddings, usually 200
    :param word_index: hash-map for mapping words to indices
    :return:
    """
    model = Fasttext.load_model(path)
    num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    not_found_words = []
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = model.get_word_vector(word)  # try to find a vector for the given word
        except KeyError:
            not_found_words.append(word)
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:num_words], axis=0)  # <UNK> token as mean of all the vectors
    if len(not_found_words) > 0:
        print('Not found embedding vectors for {} words out of {}'.format(len(not_found_words), len(word_index)))
        print(" ".join(not_found_words))
    return embedding_matrix

def glove_embeddings(path='glove.twitter.27B.200d.txt', num_words=200000, embedding_dim=200, word_index=None):
    """
    Create embedding matrix from the given GloVe embeddings
    :param word_index:
    :return:
    """
    # 1. Create GloVe word_index
    glove_word_index = {'<PAD>': 0}
    glove_vectors = []
    glove_vectors.append(np.zeros((embedding_dim,)))
    with open(path, encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            token, raw_vector = line.split(' ', maxsplit=1)
            # token = str(token, 'utf-8')
            vector = np.array([float(x) for x in raw_vector.split()])

            glove_word_index[token] = i + 1
            glove_vectors.append(vector)

    # 2. Create embedding_matrix using our vocab
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    num_words = min(num_words, len(word_index))
    not_found_words = []
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = glove_vectors[glove_word_index[word]]
        except KeyError:
            not_found_words.append(word)
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:num_words], axis=0)               # <UNK> token as mean of all the vectors
    if len(not_found_words) > 0:
        print('Not found embedding vectors for {} words out of {}'.format(len(not_found_words), len(word_index)))
        print(" ".join(not_found_words))
    return embedding_matrix

def random_embeddings(num_words=200000, embedding_dim=200, word_index=None):
    """ Create random embedding vectors for all words """
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    num_words = min(num_words, len(word_index))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    return embedding_matrix

def smn_word2vec_embedding(num_words=200000, embedding_dim=200, word_index=None):
    """
    Use prepared vectors and prepared word_index created by authors of SMN
    :param num_words:
    :param embedding_dim:
    :param word_index:
    :return:
    """
    # 1. Use prepared word_index
    smn_word_index = pickle.load(open('../helpers/smn_w2v_word_index.pkl', 'rb'), encoding='latin1')

    # 2. Use prepared vectors
    smn_vectors = pickle.load(open('../helpers/smn_w2v_embeddings.pkl', 'rb'), encoding='latin1')

    # 2. Create embedding_matrix using our vocab
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    num_words = min(num_words, len(word_index))
    not_found_words = []
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = smn_vectors[smn_word_index[word]]
        except KeyError:
            not_found_words.append(word)
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:num_words], axis=0)  # <UNK> token as mean of all the vectors
    if len(not_found_words) > 0:
        print('Not found embedding vectors for {} words out of {}'.format(len(not_found_words), len(word_index)))
        print(" ".join(not_found_words))
    return embedding_matrix

import re, os
from twokenize import tokenize
from twokenize import is_url, is_number, is_version

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.replace('`', '\'')

    # string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r"%20", " ", string)

    # string = re.sub(r"`", " ` ", string)
    # string = re.sub(r",", " , ", string)
    return string.strip()

def remove_punctuation(string):
    # as Keras filters does
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.replace(',', '')

    for i in range(2):
        # Then remove all punctuation characters
        string = re.sub(r"(?:\s|^)\.(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\"(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^):(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)'(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)!(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^);(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)](?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\[(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\](?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)}(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^){(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)`(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\?(?:\s|$)", " ", string)

        string = re.sub(r"(?:\s|^)\((?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\)(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)#(?:\s|$)", " ", string)

        # string = re.sub(r"(?:\s|^)<(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)>(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)-(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)*(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)=(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)+(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)|(?:\s|$)", " ", string)
        #
        # string = re.sub(r"(?:\s|^)[\[\])(=<>\\.*:;@/_+&~\"'-]{2}(?:\s|$)", " ", string)  # replace '--', '"?'

        # string = re.sub(r"(?:\s|^)'s(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'ll(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'d(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'ve(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'re(?:\s|$)", " ", string)


    # string = string.replace("__eot__", "%%%%%EOT%%%%%")
    # string = string.replace("__eou__", "%%%%%EOU%%%%%")
    # string = string.replace("_","")
    # string = string.replace("%%%%%EOT%%%%%"," _eot_ ")
    # string = string.replace("%%%%%EOT%%%%%", " _eou_ ")

    string = string.replace("__eot__", " _eot_ ")
    # string = string.replace("__eou__", " _eou_ ")

    return string

def process_token(c, word):
    """
    Use NLTK to replace named entities with generic tags.
    Also replace URLs, numbers, and paths.
    """
    # nodelist = ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'FACILITY', 'GSP']
    # if hasattr(c, 'label'):
    #     if c.label() in nodelist:
    #         return "__%s__" % c.label()
    if is_url(word):
        return "__URL__"
    elif is_number(word):
        return "__NUMBER__"
    elif os.path.isabs(word):
        return "__PATH__"
    return word

def process_line(s, clean_string=True):
    """
    Processes a line by iteratively calling process_token.
    """
    if clean_string:
        s = clean_str(s)
    tokens = tokenize(s)
    # sent = nltk.pos_tag(tokens)
    # chunks = nltk.ne_chunk(sent, binary=False)

    # return [process_token(c,token).lower() for c,token in zip(chunks, tokens)]
    return [process_token(None, token).lower() for token in tokens]    # do not use POS tagging


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=200000, type=int)

    psr.add_argument('--train_data', default='../ubuntu_data/train.txt')   # path to source data
    psr.add_argument('--valid_data', default='../ubuntu_data/valid.txt')
    psr.add_argument('--test_data', default='../ubuntu_data/test.txt')
    psr.add_argument('--w2v_path', default='')

    psr.add_argument('--version', default=1, type=int)  # which version of Ubuntu Dataset to use
    psr.add_argument('--sampling', default='1-1', type=str) # how to create train data for Ubuntu v3
    psr.add_argument('--embeddings', default='word2vec') # which embeddings to use

    psr.add_argument('--for_w2v', default='no')
    psr.add_argument('--dumped', default='no')
    args = psr.parse_args()

    print('load data')
    global filters
    filters = "\t\n,"

    train_context = train_response = valid_context = valid_response = None
    train_labels = valid_labels = test_labels = None
    if args.version == 1:
        train_context, train_response, train_labels = build_multiturn_data("../ubuntu_data/train.txt")
        valid_context, valid_response, valid_labels = build_multiturn_data("../ubuntu_data/valid.txt")
        test_context, test_response, test_labels = build_multiturn_data("../ubuntu_data/test.txt")
    elif args.version == 2:
        if args.dumped == 'yes':
            # load saved splitted words
            with open('../v2_joblib/prep_train_context.pickle', 'rb') as h1, \
                    open('../v2_joblib/prep_train_response.pickle', 'rb') as h2, \
                    open('../v2_joblib/prep_valid_context.pickle', 'rb') as h3, \
                    open('../v2_joblib/prep_valid_response.pickle', 'rb') as h4, \
                    open('../v2_joblib/prep_test_context.pickle', 'rb') as h5, \
                    open('../v2_joblib/prep_test_response.pickle', 'rb') as h6, \
                    open('../v2_joblib/prep_train_labels.pickle', 'rb') as h7:
                train_context, train_response, valid_context, valid_response, test_context, test_response, train_labels = \
                    pickle.load(h1), pickle.load(h2), pickle.load(h3), pickle.load(h4), pickle.load(h5), pickle.load(
                        h6), pickle.load(h7)
        else:
            train_context, train_response, train_labels = build_multiturn_data("../ubuntu_data_v2/train.csv", version=args.version)
            valid_context, valid_response, valid_labels = build_multiturn_data("../ubuntu_data_v2/valid.csv", version=args.version)
            test_context, test_response, test_labels = build_multiturn_data("../ubuntu_data_v2/test.csv", version=args.version)
            #
            # dump parsed dataset
            with open('../v2_joblib/prep_train_context.pickle', 'wb') as handle:
                 pickle.dump(train_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_train_response.pickle', 'wb') as handle:
                 pickle.dump(train_response, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_valid_context.pickle', 'wb') as handle:
                 pickle.dump(valid_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_valid_response.pickle', 'wb') as handle:
                 pickle.dump(valid_response, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_test_context.pickle', 'wb') as handle:
                pickle.dump(test_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_test_response.pickle', 'wb') as handle:
                pickle.dump(test_response, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../v2_joblib/prep_train_labels.pickle', 'wb') as handle:
                pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Debug
            with open('preptrain', 'wt') as fout:
                for line in (train_context + train_response):
                    fout.write("\n" + line)
            #
            # with open('prepvalid', 'wt') as fout:
            #     for line in (valid_context + valid_response):
            #         fout.write("\n" + line)
            #
            # with open('preptest', 'wt') as fout:
            #     for line in (test_context + test_response):
            #         fout.write("\n" + line)

    elif args.version == 3:
        # TODO: outdated!
        # because we don't have test data for DSTC7 Ubuntu Corpus v3 yet!
        # train_context, train_response, train_labels = \
        #     build_multiturn_data("../ubuntu_data_v3/ubuntu_train_subtask_1.json", args.version, "train", args.sampling)
        # valid_context, valid_response, valid_labels =\
        #     build_multiturn_data("../ubuntu_data_v3/ubuntu_dev_subtask_1.json", args.version, "valid", args.sampling)

        # or load existing pickled versions!!!
        with open('../v3_joblib/prep_train_context.pickle', 'rb') as handle:
            train_context = pickle.load(handle)
        with open('../v3_joblib/prep_train_response.pickle', 'rb') as handle:
            train_response = pickle.load(handle)
        with open('../v3_joblib/prep_valid_context.pickle', 'rb') as handle:
            valid_context = pickle.load(handle)
        with open('../v3_joblib/prep_valid_response.pickle', 'rb') as handle:
            valid_response = pickle.load(handle)
        # TODO: labels ????

    global word_index
    global stop_words
    tokenizer = None
    # tokenize
    print('tokenize')
    if args.version == 1:
        if args.dumped == 'yes':
            # load existing tokenizer for faster load
            with open('../v1_joblib/v1_tokenizer.pickle', 'rb') as handle:
                print('restoring tokenizer for v1')
                tokenizer = pickle.load(handle)
        else:
            # save tokenizer
            tokenizer = Tokenizer(filters=filters, split=" ")
            print('fit tokenizer')
            tokenizer.fit_on_texts(train_context + train_response)
            with open('../v1_joblib/v1_tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.version == 2:
        # TODO: tokenize for v2 and save tokenizer
        if args.dumped == 'yes':
            with open('../v2_joblib/v2_tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = Tokenizer(filters=filters, split=" ")
            tokenizer.fit_on_texts(train_context + train_response)
            # save tokenizer
            with open('../v2_joblib/v2_tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.version == 3:
        # TODO: outdated!
        with open('../v3_joblib/prep_train_context.pickle', 'wb') as handle:
            pickle.dump(train_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../v3_joblib/prep_train_response.pickle', 'wb') as handle:
            pickle.dump(train_response, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../v3_joblib/prep_valid_context.pickle', 'wb') as handle:
            pickle.dump(valid_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../v3_joblib/prep_valid_response.pickle', 'wb') as handle:
            pickle.dump(valid_response, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # tokenizer = Tokenizer(filters="\t\n,", split=" ")
        # tokenizer.fit_on_texts(train_context + train_response)
        # save tokenizer
        # with open('../v3_joblib/v3_tokenizer.pickle', 'wb') as handle:
        #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('../v3_joblib/v3_tokenizer.pickle', 'rb') as handle:
            print('restoring tokenizer for v3')
            tokenizer = pickle.load(handle)

    # fix bug with num_words... Keras sucks again
    all_words_count = len(tokenizer.word_index)
    tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if
                            i < args.num_words}  # <= because tokenizer is 1 indexed
    tokenizer.word_index['<UNK>'] = args.num_words
    word_index = tokenizer.word_index  # We will use word_index below for creating embedding matrix
    print('Found {} tokens; will use {} unique tokens'.format(all_words_count, len(word_index)))

    if args.for_w2v == 'yes':
        exit('finish, watch preptran')

    # Create embeddigns
    embedding_matrix = None
    stop_words = None
    if args.embeddings == 'word2vec':   # our trained word2vec
        # set the embedding file
        w2v_path = ""
        if args.version == 1:
            w2v_path = '../ubuntu_word2vec_200.model'
        elif args.version == 2:
            w2v_path = args.w2v_path
            # w2v_path = '../v2_ubuntu_word2vec_200_min_count1_iter10_window_15_sg_1.model'  # TODO: replace word2vec model here
        elif args.version == 3:
            w2v_path = '../v3_ubuntu_word2vec_200.model'

        # create embedding matrix
        print('create embedding matrix')
        embedding_matrix = word2vec_embedding(path=w2v_path,
                                              num_words=args.num_words,
                                              embedding_dim=200,
                                              word_index=word_index)

    elif args.embeddings == 'glove':
        # create embedding matrix
        w2v_path = '../glove.twitter.27B.200d.txt'
        embedding_matrix = glove_embeddings(w2v_path,
                                            num_words=args.num_words,
                                            embedding_dim=200,
                                            word_index=word_index
                                            )
    elif args.embeddings == 'random':
        # create random embedding matrix
        embedding_matrix = random_embeddings(num_words=args.num_words,
                                             embedding_dim=200,
                                             word_index=word_index
                                             )
    elif args.embeddings == 'smn_word2vec':
        # use prepared vectors and prepared word_index
        embedding_matrix = smn_word2vec_embedding(
            num_words=args.num_words,
            embedding_dim=200,
            word_index=word_index
        )
    elif args.embeddings == 'fasttext':
        emb_path = ""
        if args.version == 1:
            emb_path = None
        elif args.version == 2:
            emb_path = args.w2v_path
        elif args.version == 3:
            emb_path = None

        embedding_matrix = fasttext_embeddings(
            path=emb_path,
            num_words=args.num_words,
            embedding_dim=300,
            word_index=word_index
        )

    # 4. Use tokenizer to convert words to integer numbers
    train_context, train_context_len, word_train_context = preprocess_multi_turn_texts(train_context, args.max_turn)
    train_response, train_response_len, word_train_response = preprocess_texts(train_response)
    valid_context, valid_context_len, word_valid_context = preprocess_multi_turn_texts(valid_context, args.max_turn)
    valid_response, valid_response_len, word_valid_response = preprocess_texts(valid_response)
    if args.version in [1, 2]:
        # because we don't have test data yet!
        test_context, test_context_len, word_test_context = preprocess_multi_turn_texts(test_context, args.max_turn)
        test_response, test_response_len, word_test_response = preprocess_texts(test_response)

    # 5. We should store sentences and sentences length
    train_data_context = {'context': train_context,
                  'context_len': train_context_len}
    train_data_response = {
                  'response': train_response,
                  'response_len': train_response_len,
                  'labels': train_labels}
    valid_data_context = {'context': valid_context,
                  'context_len': valid_context_len}
    valid_data_response = {
                  'response': valid_response,
                  'response_len': valid_response_len,
                  'labels': valid_labels}
    # In DSTC7 (v3) we don't have test set
    if args.version in [1, 2]:
        # because we don't have test data yet!
        test_data_context = {'context': test_context,
                     'context_len': test_context_len}
        test_data_response = {
                     'response': test_response,
                     'response_len': test_response_len,
                     'labels': test_labels}

    print('dump embedding matrix')
    joblib.dump(embedding_matrix, 'embedding_matrix.joblib', protocol=-1, compress=3)

    if NEED_WORDS:
        # save raw words
        word_train_data = {'context': word_train_context, 'response': word_train_response }
        word_valid_data = {'context': word_valid_context, 'response': word_valid_response}
        word_test_data = {'context': word_test_context, 'response': word_test_response}

    print('dump processed data')
    joblib.dump(train_data_context, 'train_context.joblib', protocol=-1, compress=3)
    joblib.dump(train_data_response, 'train_response.joblib', protocol=-1, compress=3)

    joblib.dump(valid_data_context, 'valid_context.joblib', protocol=-1, compress=3)
    joblib.dump(valid_data_response, 'valid_response.joblib', protocol=-1, compress=3)
    if args.version in [1, 2]:
        joblib.dump(test_data_context, 'test_context.joblib', protocol=-1, compress=3)
        joblib.dump(test_data_response, 'test_response.joblib', protocol=-1, compress=3)

    if NEED_WORDS:
        # save raw words for ELMo
        joblib.dump(word_train_data, 'word_train_context.joblib', protocol=-1, compress=3)
        joblib.dump(word_valid_data, 'word_valid_context.joblib', protocol=-1, compress=3)
        joblib.dump(word_test_data, 'word_test_context.joblib', protocol=-1, compress=3)

if __name__ == '__main__': main()
