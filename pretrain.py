import argparse
from gensim.models.word2vec import Word2Vec, Text8Corpus
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import ijson
import pickle
from tqdm import tqdm
import codecs
from collections import Counter

def create_dialog_iter(filename):
    """
    Returns an iterator over a JSON file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            row = process_dialog(entry)
            yield row


def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    :param dialog:
    :return:
    """

    row = []
    utterances = dialog['messages-so-far']
    for msg in utterances:
        row.append(msg['utterance'])

    for i, utterance in enumerate(dialog['options-for-next']):
        row.append(utterance['utterance'])

    return row

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('-d', '--dim', default=200, type=int)
    psr.add_argument('-p', '--path', default='ubuntu_data/train.txt')
    psr.add_argument('-v', '--version', default=1, type=int)
    psr.add_argument('--iter', default=1, type=int)
    psr.add_argument('--window', default=5, type=int)
    args = psr.parse_args()

    if args.version == 1:
        filters = "\t\n,"
        # Parse and tokenize Ubuntu Corpus v1
        # sentences = Text8Corpus(args.path)
        sentences = []
        with codecs.open(args.path,'r','utf-8') as f:
            for line in tqdm(f):
                line = line.replace('_','')
                parts = line.strip().split('\t')

                for i in range(1, len(parts)-1, 1):
                    sentences.append(text_to_word_sequence(parts[i], filters=filters, split=" "))

                response = parts[-1]
                sentences.append(text_to_word_sequence(response, filters=filters, split=" "))

        print('training')
        model = Word2Vec(sentences, iter=1, size=args.dim, sg=1, window=5, min_count=1, workers=8)
        model.save('ubuntu_word2vec_' + str(args.dim) + '.model')
        print('saved.')
    elif args.version == 2:
        filters = "\t\n,"
        # read context and response
        print('reading')
        sentences = []
        # tokenizer = None
        # with open('v2_joblib/v2_tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle)
        with codecs.open('tf/preptrain', 'r', 'utf-8') as text_f:
            for line in text_f:
                # if len(line) > 2000: break
                line = line.replace('_eot_', '')
                sentences.append(text_to_word_sequence(line, filters=filters, split=" "))
        # print(sentences)
        print('training')
        model = Word2Vec(sentences, iter=args.iter, size=args.dim, sg=1, window=args.window, min_count=1, workers=8)
        model_filename = 'v2_ubuntu_word2vec_{}_iter{}_window_{}_sg_1_tokenization_02oct.model'\
            .format(args.dim, args.iter, args.window)
        model.save(model_filename)
        counts = Counter({word: vocab.count for (word, vocab) in model.wv.vocab.items()}); topn = 50
        print('top', topn, 'words, ', counts.most_common(topn))
        print("most similar to 'x11': {}".format(model.wv.most_similar("x11", topn=15)))
        print("most similar to 'install': {}".format(model.wv.most_similar("install", topn=15)))
        print('saved, {}'.format(model_filename))

    elif args.version == 3:
        filters = "\t\n,"
        # read context and response
        print('reading')
        sentences = []
        # tokenizer = None
        # with open('v2_joblib/v2_tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle)
        with codecs.open('tf/preptrain_v3', 'r', 'utf-8') as text_f:
            for line in text_f:
                line = line.replace('_eot_', '')
                sentences.append(text_to_word_sequence(line, filters=filters, split=" "))
        # print(sentences)
        print('training')
        model = Word2Vec(sentences, iter=args.iter, size=args.dim, sg=1, window=args.window, min_count=1, workers=8)
        model_filename = 'v3_ubuntu_word2vec_{}_iter{}_window_{}_sg_1_tokenization_02oct.model' \
            .format(args.dim, args.iter, args.window)
        model.save(model_filename)
        counts = Counter({word: vocab.count for (word, vocab) in model.wv.vocab.items()});
        topn = 50
        print('top', topn, 'words, ', counts.most_common(topn))
        print("most similar to 'x11': {}".format(model.wv.most_similar("x11", topn=15)))
        print("most similar to 'install': {}".format(model.wv.most_similar("install", topn=15)))
        print('saved, {}'.format(model_filename))

if __name__ == '__main__': main()
