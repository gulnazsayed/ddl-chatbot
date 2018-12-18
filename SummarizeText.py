# author: Gulnaz Sayed

import wikipedia


# User inputs search term and the summary of the respective wikipedia page summary is retrieved from the web.
def retrieve_text(title):
    p = wikipedia.page(title)
    document = p.summary
    document = document.replace('\n', '')
    print(document)


term = input("What would you like to search on Wikipedia? ")
retrieve_text(term)


