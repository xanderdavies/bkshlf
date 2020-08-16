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
def check_title(text, proposed_title):
    words = text.split()
    for word in words:
        if word in proposed_title:
            print("proposed title rejected")
            return False
        return True


# %% text_to_book function
# book_text: string of all the text the OCR found, separated by spaces
def text_to_book(book_text_pair):
    book = []
    for book_text in book_text_pair:
        str = '+'.join(book_text.split())
        resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
        j_resp = resp.json()
        try:
            top_result = j_resp["items"][0]
            title = top_result["volumeInfo"]["title"]
            authors = top_result["volumeInfo"]["authors"]
            description = top_result["volumeInfo"]["description"]
            id = top_result["id"]
            if check_title(book_text, title):
                book.append(Book(title, authors, id, description))
            break
        except KeyError:
            print(f"No book found matching {book_text}")
    return book


# # %% example
# sample_text = "Fodors aii GRAND CANYON 2016"
# my_book = text_to_book(sample_text)
# print(f"Book Title: {my_book.title}")
# print(f"Book Description: {my_book.description}")
