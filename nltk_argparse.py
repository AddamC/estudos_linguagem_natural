import argparse
import traceback

import nltk
from nltk.corpus import mac_morpho, brown
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import json

def processar(args):
    frase = args.sentence

    d = {'lang': args.lang,
         'sentence': args.sentence} # define os valores basicos para o json
    classificacao = {}

    arquivo = open("classificao.txt", "w") # Arquivo de saída

    mac_morpho_tagged_sents = mac_morpho.tagged_sents () # frases classificadas da mac_morpho
    mac_morpho_sents = mac_morpho.sents ()

    # classificador unigrama - UnigramTagger(), que recebe um corpo
    # de texto já classificado para treino como parâmetro para o método
    unigram_tagger_mac_morpho = nltk.UnigramTagger (mac_morpho_tagged_sents)

    frase_classificada = unigram_tagger_mac_morpho.tag (word_tokenize (frase))

    for v, k in frase_classificada:
        if v in d.keys():
            classificacao[k].append(v)
        else:
            classificacao[k] = v

    d['classificação'] = classificacao

    # string = json.dumps(d, indent=4, sort_keys=True)
    string = json.dumps(d, indent=4, ensure_ascii=False)

    print(string)


    #arquivo.write("\nmac_morpho: " + str (unigram_tagger_mac_morpho.tag (word_tokenize (frase))))
    arquivo.write(string)
    arquivo.close()

def processarIngles(frase):
    arquivo = open("classificaoEN.txt", "w")
    texto_treino = brown
    custom_sent_tokenizer = PunktSentenceTokenizer(texto_treino)
    tokenized = custom_sent_tokenizer.tokenize (frase)
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # arquivo.write(tagged)
            arquivo.write(tagged)

    except:
        traceback.print_exec()
    finally:
        arquivo.close()

def Main():
    # arquivo = open("classificacoes.json", "w")

    parser = argparse.ArgumentParser (description="processamento de linuagem natural")

    # parser.add_argument ("process", help="increase output verbosity")
    parser.add_argument ("sentence", type=str, help="value to pass as phrase")
    parser.add_argument ("lang", type=str, help="choose the language to process")

    args = parser.parse_args ()

    processar(args)

    # if args.lang == "PT":
    #     processarPortuges (frase)
    # elif args.lang == "EN":
    #     processarIngles (frase)

if __name__ == '__main__':
    Main()