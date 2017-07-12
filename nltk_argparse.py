import argparse
import traceback

import nltk
from nltk.corpus import mac_morpho, brown
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer, util

import json

def process_sentence(d):
    d_tagged_sent = {}
    file = open("tagged_sents_PT.txt", "w")  # Arquivo de saída

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
    # d['sentence polarity']

    string = json.dumps(d, indent=4, ensure_ascii=False, sort_keys=False)

    print(string)

    #arquivo.write("\nmac_morpho: " + str (unigram_tagger_mac_morpho.tag (word_tokenize (frase))))
    # file.write(string)
    # file.close()

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

def Main():
    parser = argparse.ArgumentParser (description="Natural Language Processing")

    parser.add_argument ("sentence", type=str, help="value to pass as phrase")
    parser.add_argument ("lang", type=str, help="choose the language to process")

    args = parser.parse_args ()
    dictionary = {'lang': args.lang, 'sentence': args.sentence} # define os valores basicos para o json

    process_sentence(dictionary)

if __name__ == '__main__':
    Main()
