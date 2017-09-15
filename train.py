# coding=utf-8
import numpy as np
import torch as t
import jieba
import pickle
from .word2vec import  MyChineseSentences, MySentences
# 工具函数
def get_embedding_info(lan='en'):

    input_ = 'input/'+lan
    W = np.load(input_+'word2vec.npy')
    words = pickle.load(open(input_+'vocab.pkl', "rb"))
    # 句子开头结尾
    function_words = ['PADDING', 'START', 'END', 'OOV_WORD']
    words = function_words + list(words.keys())
    _word2id = dict(zip(words, range(len(words))))

    input_dim = len(words)
    # 构建权重矩阵，初始值为0
    weights = np.zeros((input_dim, W.shape[1]), np.float32)
    weights[1] = np.ones((W.shape[1]), np.float32) * 0.33  # START
    weights[2] = np.ones((W.shape[1]), np.float32) * 0.66  # END
    weights[3] = np.average(W, axis=0)
    weights[4:] = W  # 初始化FUNCTION_WORDS以外的单词
    return _word2id, weights


def word_id(word2id, word):
    try:
        return word2id[word]
    except KeyError:
        return 3
def get_data_read_for_train(batch_size = 1024):

    en_sentences = 'input/train.en'
    zh_sentences = 'input/train.zh'
    for en, zh in zip(en_sentences, zh_sentences):
        yield [word_id(en_word2id, w) for w in en], [word_id(zh_word2id, w) for w in zh]


if __name__ == '__main__':
    en_word2id, en_embedding = get_embedding_info()
    zh_word2id, zh_embedding = get_embedding_info('zh')