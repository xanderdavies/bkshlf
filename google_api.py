# Google Books API

# notes: some books (like "mlk image celebration and word in") are easy for goodreads
# but seem not to be in google api, same with a hemingway one, so maybe we also
# check goodreads, but a pain.

# %% imports + basic setup
import requests
import json
import random

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
        if word in (proposed_title.lower()).replace("'", ""): # remove's '
            words_in_author_title += 1
        for author in proposed_authors:
            if word in author.lower():
                words_in_author_title += 1
        if words_in_author_title >= 2:
            print(f"{proposed_title} by {proposed_authors[0]} accepted")
            return True
    print(f"{proposed_title} by {proposed_authors[0]} rejected")
    return False

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

# clean up by abstacting majorly.
def text_to_book(book_text_pair):
    book = None
    for book_text in book_text_pair:
        if book == None:
            str = '+'.join(book_text.split())
            resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
            j_resp = resp.json()
            book = resp_to_book(j_resp)
            if book == None or not check_title(book_text, book.title, book.authors):
                print("trying spell check...", end=' ')
                spell = Speller()
                book_text_spelled = spell(book_text)
                print(f"{book_text_spelled}...")
                str = '+'.join(book_text_spelled.split())
                resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
                j_resp = resp.json()
                book = resp_to_book(j_resp)
                if book != None:
                    orig_text_check = check_title(book_text, book.title, book.authors)
                    auto_text_check = check_title(book_text_spelled, book.title, book.authors)
                    if not orig_text_check and not auto_text_check:
                        book = None
                print(book)
    return book


# # %% example
# sample_text = "Fodors aii GRAND CANYON 2016"
# my_book = text_to_book(sample_text)
# print(f"Book Title: {my_book.title}")
# print(f"Book Description: {my_book.description}")
