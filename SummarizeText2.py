## importing all required packages
from stanfordcorenlp import StanfordCoreNLP
import json
import logging
import re
import nltk
from nltk.tree import ParentedTree, Tree
import spacy
import en_core_web_sm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator
from subject_verb_object_extract import findSVOs, nlp
from pattern.en import conjugate
from pattern.en import tenses
from nltk.stem import WordNetLemmatizer
import wikipedia
from newspaper import Article
wordnet_lemmatizer = WordNetLemmatizer()
from nltk import word_tokenize, pos_tag, ne_chunk
from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt
import logging
from gensim.summarization import summarize
from gensim.summarization import keywords


# functions to resolve coreferences
def resolve_coreferences(text, ER, print_mentions=False):
    nlp = StanfordCoreNLP(path_or_host='http://localhost', port=9000, timeout=30000)
    props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
    # declaring all types of pronouns
    poss_pronouns = ['his', 'her', 'my', 'mine', 'our', 'ours', 'its', 'hers', 'their', 'theirs', \
                     'your', 'yours']
    noun_set = ["'", "s"]

    puncts = ['-LRB-', "'", "'s", ',']

    subject_pronouns = ['he', 'she', 'we', 'they', 'you', 'i', 'they']
    all_pronouns = ['he', 'she', 'we', 'they', 'me', 'us', \
                    'them', 'what', 'who', 'whom', 'this', 'that', 'these', \
                    'those', 'which', 'whose', 'whoever', 'whatever', 'whichever', \
                    'whomever', 'myself', 'yourself', 'itself', \
                    'ourselves', 'themselves', \
                    'my', 'mine', 'our', 'ours', 'hers', 'their', 'theirs', \
                    'your', 'yours', 'him', \
                    'his', 'her', 'himself', 'herself', 'it', 'its', 'you', 'i' \
                    ]
    result = json.loads(nlp.annotate(text, properties=props))
    coref = list(result['corefs'].items())
    # to find number of rows of mentions or number of blocks
    rows = len(coref)
    # print('rows:',rows)
    # to print out the coreferences
    for i in range(rows):
        num, mentions = coref[i]
        if print_mentions == True:
            print()
            print("CLUSTER" + str(i + 1))
            print()
            for mention in mentions:
                print(mention)
    # tokenize the content
    sent_num = len(result['sentences'])
    sents = result['sentences']
    sent_text = []
    for i in range(sent_num):
        token_num = len(sents[i]['tokens'])
        each_sent = []
        for j in range(token_num):
            each_sent.append(sents[i]['tokens'][j]['originalText'])
        sent_text.append(each_sent)
        # replace pronouns with nouns - body
    for i in range(rows):
        mentions = coref[i][1]
        noun = mentions[0]['text']
        # functions to fix noun punctuation problems
        if check_punct_noun(noun, puncts):
            noun = fix_punct_noun(noun, puncts)
        pronoun_dict = {}
        sent_dict = {}
        for j in range(1, len(mentions)):
            pronoun = mentions[j]['text'].lower()
            sent_num = mentions[j]['sentNum']
            index = mentions[j]['startIndex']

            if noun in ER and pronoun in [word.lower() for word in ER[noun]]:
                # save the ER name into a different dictionary along with sent_num, dont replace pronouns in the same sentence
                sent_dict.update({sent_num: pronoun})

            # to check if it is a pronoun & check if the noun is already present in that sentence

            if pronoun in set(all_pronouns) \
                    and noun not in set(all_pronouns):
                # check about the isrepresentative flag
                if mentions[j]['isRepresentativeMention'] == False:
                    # if pronoun already replaced with a noun in a sentence, then don't replace any other
                    # pronouns with the same noun in the same sentence.
                    if sent_num in pronoun_dict:
                        continue
                    # don't replace pronouns when noun is already in the same sentence
                    if noun in sent_text[sent_num - 1]:
                        continue
                    # don't replace pronouns when an alias of the noun is already in the same sentence
                    if sent_num in sent_dict:
                        continue

                    # check here if possessive pronoun then add noun+'s
                    if pronoun in set(poss_pronouns) and noun[-1] not in set(noun_set):
                        sent_text[sent_num - 1][index - 1] = noun + "'s"

                    else:
                        sent_text[sent_num - 1][index - 1] = noun

                    # add key('sentNum'),value(noun,pronoun) to dictionary
                    pronoun_dict.update({sent_num: (noun, pronoun)})

                    # if prounoun has more than one word, replace remaining words with blank
                    if (len(mentions[j]['text'].split()) > 1):
                        for k in range(1, len(mentions[j]['text'].split())):
                            sent_text[sent_num - 1][index - 1 + k] = ""
        # untokenizing content
        for i in range(len(sent_text)):
            sent_text[i] = untokenize(sent_text[i])
        # joining all sentences together
        text = ""
        text = sent_text[0]
        for i in range(1, len(sent_text)):
            text = text + " " + sent_text[i]

        return text

#function to check if there are punctuations in the noun of corefs
def check_punct_noun(noun,puncts):
    for punct in puncts:
        if bool(re.search(punct,noun)):
            return True
    return False

#function to fix the punctuations in the noun of corefs
def fix_punct_noun(noun,puncts):
    words =noun.split()
    noun=[]
    for word in words:
        if word not in set(puncts):
            noun.append(word)
        else:
            break

    if len(noun)>1:
        noun=untokenize(noun)
        return noun

    else:
        return noun[0]


# functions to resolve entities
def resolve_entities(text, print_mentions=False):
    nlp = StanfordCoreNLP(path_or_host='http://localhost', port=9000, timeout=30000)
    props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
    puncts = ['-LRB-', "'", "'s", ',']
    # Create ER dictionary
    ER = {}
    result = json.loads(nlp.annotate(text, properties=props))
    coref = list(result['corefs'].items())
    # to find number of rows of mentions or number of blocks
    rows = len(coref)
    # print('rows:',rows)
    # to print out the coreferences
    for i in range(rows):
        num, mentions = coref[i]
        if print_mentions == True:
            print()
            print("CLUSTER" + str(i + 1))
            print()
            for mention in mentions:
                print(mention)
    # tokenize the content
    sent_num = len(result['sentences'])
    sents = result['sentences']
    sent_text = []
    for i in range(sent_num):
        token_num = len(sents[i]['tokens'])
        each_sent = []
        for j in range(token_num):
            each_sent.append(sents[i]['tokens'][j]['originalText'])
        sent_text.append(each_sent)

    # Find NERS
    NERS = NER_all(text)
    # print(NERS)

    # replace pronouns with nouns - body
    for i in range(rows):
        mentions = coref[i][1]
        noun = mentions[0]['text']
        # functions to fix noun punctuation problems
        if check_punct_noun(noun, puncts):
            noun = fix_punct_noun(noun, puncts)

        if noun not in NERS:
            continue

        pronoun_list = []

        for j in range(1, len(mentions)):
            pronoun = mentions[j]['text']
            sent_num = mentions[j]['sentNum']
            index = mentions[j]['startIndex']

            if pronoun not in NERS:
                continue

            if pronoun not in pronoun_list:
                pronoun_list.append(pronoun)

            # update the ER dictionary with name and other names
            ER.update({noun: pronoun_list})

    return ER

# defining function for untokenizing content
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

# correcting spacing gaps in document, or when there isn't a space after a period.
def period_space_corr(s):
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s))


# function for named entity recognition
def NER_all(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    NERS = []
    for ent in doc.ents:
        NERS.append((ent.text))
    NERS = list(set(NERS))
    return NERS

#Using Subject Predicate Object triples to extract new information from text
def SVOS(corpus):
    tokens = nlp(corpus[1][:-1])
    svos = findSVOs(tokens)
    svos =[]
    for sent in corpus:
        tokens = nlp(sent[:-1])
        if len(findSVOs(tokens))>0:
            svos.append(findSVOs(tokens))
    return svos

