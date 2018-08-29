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
                if sampling == '1-1':
                    rows.append(true)
                    rows.append((message, np.random.choice(fake_responses, 1), 0))  # random 1 from 99
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

def build_multiturn_data(multiturn_data, version=1, mode="train", sampling='10-10'):
    """
    Parse the source dataset files and split into context/response/train

    multiturn_data - text file to process (for example, train.txt)
    version        - if the source file is Ubuntu Dialogue Corpus v1 / v2
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

                lable = int(parts[0])
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


def preprocess_texts(texts, maxlen, word_index):
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

    def word2id(c):
        if c in word_index:
            return word_index[c]
        else:
            return word_index['<UNK>']   # return last index, usually <UNK> or mean of all words

    # sequences = tokenizer.texts_to_sequences(texts) TODO: old code
    sequences = [list(map(word2id, text_to_word_sequence(each_utt))) for each_utt in texts]
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return pad_sequences(sequences, maxlen=maxlen, padding="post"), sequences_length


def preprocess_multi_turn_texts(context, max_turn, maxlen, word_index):
    print('Trim or pad to max_turn utterances')
    multi_turn_texts = []
    for i in tqdm(range(len(context))):
        multi_turn_texts.append(context[i].split('_eot_')[-(max_turn+1):-1])
        if len(multi_turn_texts[i]) <= max_turn:
            tmp = multi_turn_texts[i][:]
            multi_turn_texts[i] = [' '] * (max_turn - len(multi_turn_texts[i]))
            multi_turn_texts[i].extend(tmp)

    print('Tokenize and pad the sentences')
    tokenized_multi_turn_texts = []
    sequences_length = []
    for i in tqdm(range(len(multi_turn_texts))):
        tokenized_sentences, tokenized_sentences_lengths = preprocess_texts(multi_turn_texts[i], maxlen, word_index)
        tokenized_multi_turn_texts.append(tokenized_sentences)
        sequences_length.append(tokenized_sentences_lengths)
    return tokenized_multi_turn_texts, sequences_length

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
            embedding_matrix[i] = np.random.uniform(-0.6, 0.6, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:], axis=0)               # <UNK> token as mean of all the vectors
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
    c = 0
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = glove_vectors[glove_word_index[word]]
        except KeyError:
            c += 1
            embedding_matrix[i] = np.random.uniform(-0.6, 0.6, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:], axis=0)               # <UNK> token as mean of all the vectors
    if c > 0:
        print('Not found embedding vectors for {} words out of {}'.format(c, len(word_index)))
    return embedding_matrix

def random_embeddings(num_words=200000, embedding_dim=200, word_index=None):
    """ Create random embedding vectors for all words """
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    num_words = min(num_words, len(word_index))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_matrix[i] = np.random.uniform(-0.6, 0.6, embedding_dim)
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
    smn_word_index = pickle.load(open('helpers/smn_w2v_word_index.pkl', 'rb'), encoding='latin1')

    # 2. Use prepared vectors
    smn_vectors = pickle.load(open('helpers/smn_w2v_embeddings.pkl', 'rb'), encoding='latin1')

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
            c += 1
            embedding_matrix[i] = np.random.uniform(-0.6, 0.6, embedding_dim)
    embedding_matrix[-1] = np.mean(embedding_matrix[1:], axis=0)  # <UNK> token as mean of all the vectors
    if len(not_found_words) > 0:
        print('Not found embedding vectors for {} words out of {}'.format(len(not_found_words), len(word_index)))
        print(" ".join(not_found_words))
    return embedding_matrix


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=200000, type=int)

    psr.add_argument('--train_data', default='./ubuntu_data/train.txt')   # path to source data
    psr.add_argument('--valid_data', default='./ubuntu_data/valid.txt')
    psr.add_argument('--test_data', default='./ubuntu_data/test.txt')

    psr.add_argument('--version', default=1, type=int)  # which version of Ubuntu Dataset to use
    psr.add_argument('--sampling', default='1-1', type=str) # how to create train data for Ubuntu v2
    psr.add_argument('--embeddings', default='word2vec') # which embeddings to use
    args = psr.parse_args()

    print('load data')
    if args.version == 1:
        train_context, train_response, train_labels = build_multiturn_data("./ubuntu_data/train.txt")
        valid_context, valid_response, valid_labels = build_multiturn_data("./ubuntu_data/valid.txt")
        test_context, test_response, test_labels = build_multiturn_data("./ubuntu_data/test.txt")
    else:
        # because we don't have test data for Ubuntu Corpus v2 yet!
        train_context, train_response, train_labels = \
            build_multiturn_data("ubuntu_data_v2/ubuntu_train_subtask_1.json", args.version, "train", args.sampling)
        valid_context, valid_response, valid_labels =\
            build_multiturn_data("ubuntu_data_v2/ubuntu_dev_subtask_1.json", args.version, "valid", args.sampling)

    embedding_matrix = None
    tokenizer = None
    # tokenize
    print('tokenize')
    if args.version == 1:
        # load existing tokenizer for faster load
        with open('tokenizer.pickle', 'rb') as handle:
            print('restoring tokenizer for v1')
            tokenizer = pickle.load(handle)
        # tokenizer.fit_on_texts(np.append(train_context, train_response)) # TODO: numpy can throw MemoryError here
    elif args.version == 2:
        # tokenizer = Tokenizer(filters="\t\n,", split=' ')
        # sentences = Text8Corpus("ubuntu_data_v2/train.txt")
        # tokenizer.fit_on_texts(sentences)
        with open('v2_joblib/v2_tokenizer.pickle', 'rb') as handle:
            print('restoring tokenizer for v2')
            tokenizer = pickle.load(handle)

    # save for faster access
    # with open('tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # fix bug with num_words... keras sucks again
    all_words_count = len(tokenizer.word_index)
    tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if
                            i < args.num_words}  # <= because tokenizer is 1 indexed
    tokenizer.word_index['<UNK>'] = args.num_words
    word_index = tokenizer.word_index  # We will use word_index below for creating embedding matrix
    print('Found {} tokens; will use {} unique tokens'.format(all_words_count, len(word_index)))
    if args.embeddings == 'word2vec':
        # set the embedding file
        w2v_path = 'ubuntu_word2vec_200.model' if args.version == 1 else 'v2_ubuntu_word2vec_200.model'

        # create embedding matrix
        print('create embedding matrix')
        embedding_matrix = word2vec_embedding(path=w2v_path,
                                              num_words=args.num_words,
                                              embedding_dim=200,
                                              word_index=word_index)

    elif args.embeddings == 'glove':
        # create embedding matrix
        w2v_path = 'glove.twitter.27B.200d.txt'
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

    # 4. Use tokenizer to convert words to integer numbers
    train_context, train_context_len = \
        preprocess_multi_turn_texts(train_context, args.max_turn, args.maxlen, word_index)
    train_response, train_response_len = \
        preprocess_texts(train_response, args.maxlen, word_index)
    valid_context, valid_context_len = \
        preprocess_multi_turn_texts(valid_context, args.max_turn, args.maxlen, word_index)
    valid_response, valid_response_len = \
        preprocess_texts(valid_response, args.maxlen, word_index)
    if args.version == 1:
        # because we don't have test data yet!
        test_context, test_context_len = \
            preprocess_multi_turn_texts(test_context, args.max_turn, args.maxlen, word_index)
        test_response, test_response_len = \
            preprocess_texts(test_response, args.maxlen, word_index)


    train_data = {'context': train_context, 'response': train_response, 'labels': train_labels}
    valid_data = {'context': valid_context, 'response': valid_response, 'labels': valid_labels}
    if args.version == 1:
        test_data = {'context': test_context, 'response': test_response, 'labels': test_labels}

    print('dump')
    dump_path = "" if args.version == 1 else "v2_joblib/"
    joblib.dump(train_data, dump_path+'train.joblib', protocol=-1, compress=3)
    joblib.dump(valid_data, dump_path+'valid.joblib', protocol=-1, compress=3)
    if args.version == 1:
        joblib.dump(test_data, dump_path+'test.joblib', protocol=-1, compress=3)
    joblib.dump(embedding_matrix, dump_path+'embedding_matrix.joblib', protocol=-1, compress=3)

if __name__ == '__main__': main()
