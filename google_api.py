# Google Books API

# %% imports + basic setup
import requests
import json
APIKEY = "AIzaSyCQCfV4eIoFOdWkXClJtPJYqWMU0Gds9RE"

# %% book class
class Book:
    def __init__(self, title, authors, id, description):
        self.title = title
        self.authors = authors
        self.id = id
        self.description = description

# %% helper for text_to_book, rejects title if no book_text words
def check_title(text, proposed_title, proposed_authors):
    words = text.split()
    words_in_author_title = 0
    for word in words:
        if word in proposed_title.lower():
            words_in_author_title += 1
        for author in proposed_authors:
            if word in author.lower():
                words_in_author_title += 1
    enough_words = (words_in_author_title >= 2)
    if enough_words:
        print(f"{proposed_title} by {proposed_authors[0]} accepted")
    else:
        print(f"{proposed_title} by {proposed_authors[0]} rejected")
    return enough_words

# text_to_book_helper
def resp_to_book(json_resp):
    try:
        top_result = json_resp["items"][0]
        title = top_result["volumeInfo"]["title"]
        authors = top_result["volumeInfo"]["authors"]
        description = top_result["volumeInfo"]["description"]
        id = top_result["id"]
        return Book(title, authors, id, description)
    except KeyError:
        return None

# %% text_to_book function
# book_text: string of all the text the OCR found, separated by spaces
from autocorrect import Speller
# !pip install autocorrect

def text_to_book(book_text_pair):
    for book_text in book_text_pair:
        str = '+'.join(book_text.split())
        resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
        j_resp = resp.json()
        book = resp_to_book(j_resp)
        if book == None or not check_title(book_text, book.title, book.authors):
            print("trying spell check...")
            spell = Speller()
            book_text = spell(book_text)
            print(f"trying {book_text}...")
            str = '+'.join(book_text.split())
            resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
            j_resp = resp.json()
            book = resp_to_book(j_resp)
            if book != None:
                if not check_title(book_text, book.title, book.authors):
                    book = None
            print(book)
        return book


# # %% example
# sample_text = "Fodors aii GRAND CANYON 2016"
# my_book = text_to_book(sample_text)
# print(f"Book Title: {my_book.title}")
# print(f"Book Description: {my_book.description}")
