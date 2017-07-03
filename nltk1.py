from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import nltk.corpus
from nltk.corpus import floresta
from nltk import chunk
import json

train_text = nltk.corpus.mac_morpho
stopwords = nltk.corpus.stopwords.words("portuguese")

frase = "Uma bola azul"
frase2 = "Um pássaro de bico amarelo"

print("Informe uma frase de seu gosto: ")
frase3 = input()
texto = [frase, frase2, frase3]

frase4 = ""

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# def process_content():
#     try:
#         for i in tokenized:
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             print(tagged)

def process_content():
    tagged = []

    for i in custom_sent_tokenizer.tokenize(frase3):
        words = nltk.word_tokenize(i)
        tagged += nltk.pos_tag(words)
    tree = chunk.ne_chunk(tagged)
    tree.draw()

def filtrar(frase_ex, frase_final):
    for w in word_tokenize(frase_ex):
        if w not in stopwords and len(w) > 3:
            frase_final = frase_final + " " + w
    return frase_final

if __name__ == '__main__':
    # for i in texto:
    #     print(filtrar(i, frase4))
        # tokenized = custom_sent_tokenizer.tokenize(filtrar(i, frase4))
        # process_content()
    process_content ()