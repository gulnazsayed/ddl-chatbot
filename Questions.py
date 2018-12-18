# author: Gulnaz Sayed
# built from ATAP Chatbot Chapter

import spacy
from spacy import displacy

spacy_nlp = spacy.load("en")

def plot_displacy_tree(sent):
    doc = spacy_nlp(sent)
    displacy.serve(doc, style='dep')


from nltk.parse.stanford import StanfordParser

stanford_parser = StanfordParser(
    model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
)

def print_stanford_tree(sent):
    """
    Use Stanford pretrained model to extract dependency tree for use by other methods
    Returns a list of trees
    """
    parse = stanford_parser.raw_parse(sent)
    return list(parse)