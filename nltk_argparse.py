import numpy as np
import argparse
import json
import nltk
import re


from nltk.corpus import mac_morpho
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import OrderedDict

def process_sentence(d):
    d_tagged_sent = {}
    file = open("sentence_analyse_json.txt", "w")  # Arquivo de saída

    if d["lang"] == "PT":
        tagged_sent = process_portuguese(d) # tupla da frase classificada
    elif d["lang"] == "EN":
        tagged_sent = process_english(d)

    for v, k in tagged_sent:
        if v in d.keys():
            d_tagged_sent[k].append(v)
        else:
            d_tagged_sent[k] = v

    d['tagged sentence'] = d_tagged_sent

    # sentiment analysis
    check_negative_words(d)
    check_positive_words(d)

    # sentence polarity
    positive_count = d.get('positive analysis').get('positive counter')
    negative_count = d.get('negative analysis').get('negative counter')
    d['sentence polarity'] = positive_count/negative_count

    string = json.dumps(OrderedDict(d), indent=4)

    print(string)

    file.write(string)
    file.close()

def check_negative_words(d):
    ps = PorterStemmer()

    negative_words = ['bad', 'horrible', 'death', 'suffer', 'pain', 'awfull', 'error', 'fail', 'poor']
    words = word_tokenize(d.get('sentence'))

    np_negative_words = [ps.stem(w) for w in np.array(negative_words)]
    np_words = [ps.stem(w) for w in np.array(words)]

    dict_negative_words = {}

    # print(np_negative_words)
    # print(np_words)

    negative_counter = 0
    list_negative = []

    for w in np_words:
        for neg_w in np_negative_words:
            if w == neg_w:
                negative_counter += 1
                list_negative.append(w)

    dict_negative_words['negative counter'] = negative_counter
    dict_negative_words['negative words'] = list_negative

    negative_perc = (negative_counter/len(np_words)) * 100
    dict_negative_words['negative percentage'] = round(negative_perc, 2)

    d['negative analysis'] = dict_negative_words

def check_positive_words(d):
    ps = PorterStemmer()

    positive_words = ['good', 'tasty', 'best', 'success', 'happy', 'nice', 'delicious']
    words = word_tokenize(d.get('sentence'))

    np_positive_words = [ps.stem(w) for w in np.array(positive_words)]
    np_words = [ps.stem(w) for w in np.array(words)]

    dict_positive_words = {}

    # print(np_positive_words)
    # print(np_words)

    positive_counter = 0
    list_positive = []

    for w in np_words:
        for neg_w in np_positive_words:
            if w == neg_w:
                positive_counter += 1
                list_positive.append(w)

    dict_positive_words['positive counter'] = positive_counter
    dict_positive_words['positive words'] = list_positive

    positive_perc = (positive_counter/len(np_words)) * 100
    dict_positive_words['positive percentage'] = round(positive_perc, 2)

    d['positive analysis'] = dict_positive_words

def process_portuguese(d):
    mac_morpho_tagged_sents = mac_morpho.tagged_sents () # frases classificadas da mac_morpho

    # classificador unigrama - UnigramTagger(), que recebe um corpo
    # de texto já classificado para treino como parâmetro para o método
    unigram_tagger_mac_morpho = nltk.UnigramTagger (mac_morpho_tagged_sents)

    tagged_sentence = unigram_tagger_mac_morpho.tag(word_tokenize(d["sentence"]))

    return tagged_sentence

def process_english(d):
    tagged_sentence = nltk.pos_tag(word_tokenize(d["sentence"]))
    return tagged_sentence

def process_generic(training_set, d):

    return

def Main():
    parser = argparse.ArgumentParser (description="Natural Language Processing")

    parser.add_argument ("sentence", type=str, help="value to pass as phrase")
    parser.add_argument ("lang", type=str, help="choose the language to process")

    args = parser.parse_args ()
    dictionary = {'sentence': args.sentence, 'lang': args.lang} # define os valores basicos para o json

    process_sentence(dictionary)

if __name__ == '__main__':
    Main()
