from gensim.models import Word2Vec
from tqdm import tqdm
import re
import pickle
import jieba
import numpy as np
re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one


def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in tqdm(open(self.dirname)):
            yield simple_toks(line)


class MyChineseSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in tqdm(open(self.dirname, encoding='utf-8')):
            yield list(jieba.cut(line))
def train_save_wordvec_export(lan ='en'):
    output = 'input/'+lan
    if lan == 'en':
        corpus = MySentences('input/train.en')
    else:
        corpus = MyChineseSentences('input/train.zh')
    model = Word2Vec(corpus, min_count=10, sg=1,workers=2,iter=1)
    print('model trained!')
    vocabulary = model.wv.vocab
    embeddings = []
    for word in vocabulary:
        embeddings.append(model[word])
    embeddings = np.array(embeddings, dtype=np.float32)
    pickle.dump(vocabulary, open(output+'vocab.pkl', "wb"))
    np.save(output+'word2vec.npy', embeddings)
    print('word vector saved!')

if __name__ == '__main__':
    # train_save_wordvec_export(lan ='en')
    train_save_wordvec_export(lan ='zh')