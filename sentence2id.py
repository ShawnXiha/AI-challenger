# coding: utf-8
import numpy as np
import pickle
from word2vec import  MyChineseSentences, MySentences
from keras.preprocessing.sequence import pad_sequences
# 工具函数


def get_word(lan='en'):
    input_ = 'input/'+lan
    words = pickle.load(open(input_+'vocab.pkl', "rb"))
    return words


def word_id(word2id, word):
    return word2id.get(word, 3)


def sentence_to_id_list(word2id, sentence):
    return [1]+[word_id(word2id, w) for w in sentence]+[2]
    

def get_data_read_for_train(word2id, max_len=30, lan='en'):
    if lan == 'en':
        sentences = MySentences('input/train.en')
    else:
        sentences = MyChineseSentences('input/train.zh')
    output = 'input/train_' + lan + '_id_list.npy'
    id_lists = [sentence_to_id_list(word2id, sentence) for sentence in sentences]
    id_lists = np.array(pad_sequences(id_lists, maxlen=max_len), dtype=np.int32)
    np.save(output, id_lists)
    print('file saved in %s!' % output)    


if __name__ == '__main__':
    en_word2id = get_word(lan='en')
    get_data_read_for_train(en_word2id)
    zh_word2id = get_word(lan='zh')
    get_data_read_for_train(zh_word2id, lan='zh')

