# District Data Labs
# Text Summarization & Question-Answering Chatbot Model
# Author: Gulnaz Sayed

# nltk.download('stopwords')
# nltk.download('punkt')
import wikipedia
import tensorflow as tf
from stanfordcorenlp import StanfordCoreNLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer

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

# Basic model for extractive text summarization
# Ranks the importance of the sentences and then builds the summary using the most important sentences
def basic_extractive(text):
    # stemmer is used to get the root word (ex. generously --> generous)
    stem = SnowballStemmer("english")

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stop_words:
            continue

        word = stem.stem(word)

        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentence_value = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq
                else:
                    sentence_value[sentence] = freq

    sum_values = 0
    for sentence in sentence_value:
        sum_values += sentence_value[sentence]

    # Average value of a sentence from original text
    average = int(sum_values / len(sentence_value))

    summary = ''
    for sentence in sentences:
        if (sentence in sentence_value) and (sentence_value[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary

# QUESTION PARSING AND ANSWER GENERATION MODEL
# Currently in separate file for testing purposes

term = input("What would you like to search on Wikipedia? ")
doc = retrieve_text(term)
extractive_summary = basic_extractive(doc)