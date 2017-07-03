import nltk
from nltk.corpus import mac_morpho, floresta
from nltk.tokenize import word_tokenize

# são dois corpos da linguagem portuguesa(BR): mac_morpho e floresta treebank
# ambos apresentam situações do porugues apresentando frases e palavras
# além disso, ambos apresentam seus conteudos ja classificados
# sendo, assim, uma boa fonte de treinamento de classificadores


# O método tagged_sents() ou tagged_words() de cada corpo
#  retornam as classificações de seus conteúdos

mac_morpho_tagged_sents = mac_morpho.tagged_sents()
mac_morpho_sents = mac_morpho.sents()

floresta_tagged_sents = floresta.tagged_sents()
floresta_sents = floresta.sents()

unigram_tagger_mac_morpho = \
    nltk.UnigramTagger(mac_morpho_tagged_sents)    # Metdo para treino de unigramas do mac_morpho

unigram_tagger_floresta = \
    nltk.UnigramTagger(floresta_tagged_sents)   # Metdo para treino de unigramas do floresta

frase_teste1 = "Uma bola azul"
frase_teste2 = "Um pássaro de bico amarelo"

frases = []
frases_classificadas = []

# imprimir as frases_teste 1 e 2 classificadas
print(frase_teste1)
print("mac_morpho: " + str(unigram_tagger_mac_morpho.tag(word_tokenize(frase_teste1))))
print("floresta: " + str(unigram_tagger_floresta.tag(word_tokenize(frase_teste1))))

print("\n" + frase_teste2)
print("mac_morpho: " + str(unigram_tagger_mac_morpho.tag(word_tokenize(frase_teste2))))
print("floresta: " + str(unigram_tagger_floresta.tag(word_tokenize(frase_teste2))))

# parte para testar diferentes tipos de textos numa qtde informada
qtde_frases = input("\n\n" + "informe qtde de frases que deseja testar: ")

for i in range(int(qtde_frases)):
    frases.append(input("\n" + "frase: "))

for i in frases:
    frases_classificadas.append(unigram_tagger_mac_morpho.tag(nltk.word_tokenize(i)))
    frases_classificadas.append(unigram_tagger_floresta.tag(nltk.word_tokenize(i)))

for i in frases_classificadas:
        print (i)