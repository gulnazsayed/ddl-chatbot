# author: Gulnaz Sayed
# built from ATAP Chatbot Chapter

import spacy
from spacy import displacy

# Required first: python -m spacy download en
spacy_nlp = spacy.load("en")

def plot_displacy_tree(sent):
    doc = spacy_nlp(sent)
    displacy.serve(doc, style='dep')