# District Data Labs
# Text Summarization & Question-Answering Chatbot Model
# Author: Gulnaz Sayed

import wikipedia


# User inputs search term and the summary of the respective Wikipedia page summary is retrieved from the web.
# Returns the summary of the resulting Wikipedia page.
def retrieve_text(title):
    p = wikipedia.page(title)
    document = p.summary
    document = document.replace('\n', '')
    return document

# 


term = input("What would you like to search on Wikipedia? ")
text = retrieve_text(term)


