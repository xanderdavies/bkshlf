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

# %% text_to_book function
# book_text: string of all the text the OCR found, separated by spaces
def text_to_book(book_text):
    str = '+'.join(book_text.split())
    resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={APIKEY}")
    j_resp = resp.json()
    top_result = j_resp["items"][0]
    title = top_result["volumeInfo"]["title"]
    authors = top_result["volumeInfo"]["authors"]
    description = top_result["volumeInfo"]["description"]
    id = top_result["id"]
    return Book(title, authors, id, description)

# %% example
sample_text = "Fodors aii GRAND CANYON 2016"
my_book = text_to_book(sample_text)
print(f"Book Title: {my_book.title}")
print(f"Book Description: {my_book.description}")
