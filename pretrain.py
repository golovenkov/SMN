import argparse
from gensim.models.word2vec import Word2Vec, Text8Corpus
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import ijson
import pickle
import tqdm
import codecs

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
    args = psr.parse_args()
    if args.version == 1:
        # Parse and tokenize Ubuntu Corpus v1
        sentences = Text8Corpus(args.path)
        print('training')
        model = Word2Vec(sentences, size=args.dim, window=5, min_count=0, workers=8)
        model.save('ubuntu_word2vec_' + str(args.dim) + '.model')
        print('saved.')
    elif args.version == 2:
        # read context and response
        print('reading')
        filters = "\t\n,"
        sentences = []
        # tokenizer = None
        # with open('v2_joblib/v2_tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle)
        with codecs.open('tf/preptrain', 'r', 'utf-8') as text_f:
            for line in text_f:
                # if len(line) > 2000: break
                line = line.replace('__eot__', '')
                sentences.append(text_to_word_sequence(line, filters=filters, split=" "))
        # print(sentences)
        print('training')
        model = Word2Vec(sentences, iter=5, size=args.dim, sg=1, window=10, min_count=1, workers=8)
        model.save('v2_ubuntu_word2vec_' + str(args.dim) + '_min_count1_iter5_window_10_sg_1_not_aggressive_tokenization.model')
        print('saved')

    elif args.version == 3:
        # TODO: outdated!
        # Parse and tokenize Ubuntu Corpus v3
        print('reading json and dump sentences')
        fout_filename = "ubuntu_data_v2/train.txt"
        with open(fout_filename, 'wt') as fout:
            for sentence in create_dialog_iter(args.path):
                fout.write(" ".join(sentence))
        print('creating w2v')
        sentences = Text8Corpus(fout_filename)
        print('training')
        model = Word2Vec(sentences, size=args.dim, window=5, min_count=0, workers=8)
        model.save('v3_ubuntu_word2vec_' + str(args.dim) + '.model')
        print('saved.')


if __name__ == '__main__': main()
