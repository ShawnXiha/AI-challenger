from gensim.models import Word2Vec
from multiprocessing import cpu_count
from tqdm import tqdm
import re
import pickle
import jieba
import numpy as np
re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one


embedding_dim = 300

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
    model = Word2Vec(corpus, size=embedding_dim, min_count=10, sg=1, workers=cpu_count())
    print('model trained!')
    vocabulary = model.wv.vocab
    words = list(vocabulary.keys())
    # 句子开头结尾
    function_words = ['PADDING', 'START', 'END', 'OOV_WORD']
    words = function_words + words
    _word2id = dict(zip(words, range(len(words))))

    input_dim = len(words)
    embeddings = []
    for word in vocabulary:
        embeddings.append(model[word])
    embeddings = np.array(embeddings, dtype=np.float32)
    weights = np.zeros((input_dim, embedding_dim), np.float32)
    weights[1] = np.ones(embedding_dim, np.float32) * 0.33  # START
    weights[2] = np.ones(embedding_dim, np.float32) * 0.66  # END
    weights[3] = np.average(embeddings, axis=0)
    weights[4:] = embeddings  # 初始化FUNCTION_WORDS以外的单词
    pickle.dump(_word2id, open(output+'vocab.pkl', "wb"))
    np.save(output+'word2vec.npy', embeddings)
    print('word vector saved!')

if __name__ == '__main__':
    train_save_wordvec_export(lan='en')
    train_save_wordvec_export(lan='zh')