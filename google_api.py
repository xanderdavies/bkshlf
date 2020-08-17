# Google Books API

# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

# %% imports + basic setup
import requests
import json
import random

google_APIKEY = "AIzaSyCQCfV4eIoFOdWkXClJtPJYqWMU0Gds9RE"
isbn_header = {'Authorization': '44417_17eecbc4d201aa6115a43f4ddff496a4'}
# %% book class
class Book:
    def __init__(self, title, authors, id, description, publisher, image):
        self.title = title
        self.authors = authors
        self.id = id
        self.description = description
        self.publisher = publisher
        self.image = image

# %% helper for text_to_book, rejects title if no book_text words
def check_title(text, book_list):
    decent_choice = None
    for book in book_list:
        words = text.split()
        words_in_title = 0
        words_in_author = 0
        words_in_pub = 0
        for word in words:
            in_author = False
            for author in book.authors:
                if word in (author.lower()).replace("'", ""):
                    words_in_author += 1
                    in_author = True
                    break
            if not in_author:
                if word in (book.title.lower()).replace("'", ""): # remove's '
                    words_in_title += 1
                elif word in (book.publisher.lower()).replace("'", ""):
                    words_in_pub += 1
        if (words_in_author + words_in_title) >= 2 and (words_in_author >= 1 or words_in_pub >= 1):
            print(f"{book.title} by {book.authors[0]} accepted")
            return book
        if (1_words_in_author + words_in_title) >= 2:
            if decent_choice == None:
                print(f"{book.title} by {book.authors[0]} is a decent option")
                decent_choice = book
        print(f"{book.title} by {book.authors} rejected")
    return decent_choice

# text_to_book_helper
# def resp_to_book_google(json_resp):
#     try:
#         top_result = json_resp["items"][0]
#         title = top_result["volumeInfo"]["title"]
#         authors = top_result["volumeInfo"]["authors"]
#         description = top_result["volumeInfo"]["description"]
#         id = top_result["id"]
#         return Book(title, authors, id, description)
#     except KeyError:
#         return None

def resp_to_book_isbn(json_resp):
    book_list = []
    for i in range(10):
        try:
            result = json_resp["data"][i]
            title = result["title"]
            id = result["isbn13"]
            authors = result["authors"]
            description = result["synopsys"]
            publisher = result["publisher"]
            image = result["image"]
            book_list.append(Book(title, authors, id, description, publisher, image))
        except KeyError:
            return book_list
        except IndexError:
            return book_list
    return book_list

# %% text_to_book function
# book_text: string of all the text the OCR found, separated by spaces
from autocorrect import Speller
# !pip install autocorrect
# !pip install requests

# clean up by abstacting majorly.
def text_to_book(book_text_pair):
    book = None
    for book_text in book_text_pair:
        if book == None:
            str = '%20'.join(book_text.split())
            resp = requests.get(f"https://api2.isbndb.com/search/books?text={str}", headers = isbn_header)
            # str = '+'.join(book_text.split())
            # resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={google_APIKEY}")
            j_resp = resp.json()
            books = resp_to_book_isbn(j_resp)
            book = check_title(book_text, books)
            if book != None:
                return book
            else:
                print("trying spell check...", end=' ')
                spell = Speller()
                book_text_spelled = spell(book_text)
                print(f"{book_text_spelled}...")
                str = '%20'.join(book_text.split())
                resp = requests.get(f"https://api2.isbndb.com/search/books?text={str}", headers = isbn_header)
                # str = '+'.join(book_text_spelled.split())
                # resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={google_APIKEY}")
                j_resp = resp.json()
                books = resp_to_book_isbn(j_resp)
                if books != []:
                    orig_book = check_title(book_text, books)
                    auto_book = check_title(book_text_spelled, books)
                    if orig_book != None:
                        return orig_book
                    if auto_book != None:
                        return auto_book
    return None


# # %% example
# sample_text = "Fodors aii GRAND CANYON 2016"
# my_book = text_to_book(sample_text)
# print(f"Book Title: {my_book.title}")
# print(f"Book Description: {my_book.description}")
