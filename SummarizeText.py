# District Data Labs
# Text Summarization & Question-Answering Chatbot Model
# Author: Gulnaz Sayed

import wikipedia
import tensorflow as tf
from stanfordcorenlp import StanfordCoreNLP
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

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
# rank the importance of the sentences and then build the summary using the most important sentences
def temp(text):
    # creates an array of stopwords and array of all of the words in the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # a dictionary of the frequency of the words NOT in the stopwords set
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # assigns a score to every sentence in the text
    # scores are stored in a dictionary
    # since longer sentences will have an advantage over shorter sentences, divide score by number of words in sentence
    sentences = sent_tokenize(text)
    print(sentences)
    print(len(sentences))
    sentenceValue = dict()
    for sentence in sentences:
        for wordValue in freqTable:
            if wordValue[0] in sentence.lower():
                if sentence[:3] in sentenceValue:
                    sentenceValue[sentence[:3]] += wordValue[1]
                else:
                    sentenceValue[sentence[:3]] = wordValue[1]

    # finding the threshold
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    # sorts the sentences in order for summary
    summary = ''
    for sentence in sentences:
        if sentence[:12] in sentenceValue and sentenceValue[sentence[:12]] > (1.5 * average):
            summary += " " + sentence

    print(summary)

# QUESTION PARSING AND ANSWER GENERATION MODEL
#


term = input("What would you like to search on Wikipedia? ")
text = retrieve_text(term)
text = prepareText(text)
temp(text)