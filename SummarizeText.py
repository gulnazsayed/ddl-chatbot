# District Data Labs
# Text Summarization & Question-Answering Chatbot Model
# Author: Gulnaz Sayed

import wikipedia
import tensorflow as tf
from stanfordcorenlp import StanfordCoreNLP
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
# import skipthoughts

# User inputs search term and the summary of the respective Wikipedia page summary is retrieved from the web.
# Returns the summary of the resulting Wikipedia page.
def retrieve_text(title):
    p = wikipedia.page(title)
    document = p.summary
    document = document.replace('\n', '')
    return document

# Prepares the text, removing unnecessary spacing and characters.
def prepareText(document):
    # TO-DO
    return document

# ABSTRACTIVE SUMMARIZATION MODEL
# This method tokenizes all of the sentences in the Wikipedia summary.
def sentence_tokenize():
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

def encode():
    # You would need to download pre-trained models first
    model = skipthoughts.load_model()

    encoder = skipthoughts.Encoder(model)
    encoded = encoder.encode(sentences)


# Basic model of extractive summarization
# Ranks the importance of the sentences and then builds the summary using the most important sentences
def extractive_summarization(text):
    # stemmer is used to get the root word (ex. generously --> generous)
    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # print((words))

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue

        word = stemmer.stem(word)

        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    print(summary)

# QUESTION PARSING AND ANSWER GENERATION MODEL


term = input("What would you like to search on Wikipedia? ")
doc = retrieve_text(term)
temp(doc)