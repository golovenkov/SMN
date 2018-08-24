import argparse
from gensim.models.word2vec import Word2Vec, Text8Corpus
from gensim.models.word2vec import LineSentence
import ijson

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
    psr.add_argument('-v', '--version', default='1')
    args = psr.parse_args()
    if args.version == '1':
        # Parse and tokenize Ubuntu Corpus v1
        sentences = Text8Corpus(args.path)
        print('training')
        model = Word2Vec(sentences, size=args.dim, window=5, min_count=0, workers=8)
        model.save('ubuntu_word2vec_' + str(args.dim) + '.model')
        print('saved.')
    else:
        # Parse and tokenize Ubuntu Corpus v2
        print('reading json and dump sentences')
        fout_filename = "ubuntu_data_v2/train.txt"
        with open(fout_filename, 'wt') as fout:
            for sentence in create_dialog_iter(args.path):
                fout.write(" ".join(sentence))
        print('creating w2v')
        sentences = Text8Corpus(fout_filename)
        print('training')
        model = Word2Vec(sentences, size=args.dim, window=5, min_count=0, workers=8)
        model.save('v2_ubuntu_word2vec_' + str(args.dim) + '.model')
        print('saved.')


if __name__ == '__main__': main()
