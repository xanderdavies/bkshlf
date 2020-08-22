import re
import requests
import json
from utils import in_string_ish

isbn_header = {'Authorization': '44417_17eecbc4d201aa6115a43f4ddff496a4'}
google_APIKEY = "AIzaSyCQCfV4eIoFOdWkXClJtPJYqWMU0Gds9RE"

class Book:
    def __init__(self, title, authors, id, description, publisher, image):
        self.title = title
        self.authors = authors
        self.id = id
        self.description = description
        self.publisher = publisher
        self.image = image

# helper for text_to_book, goes from text -> top 5 google results
def text_to_book_list_google(text):
    str = '+'.join(text.split())
    resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={google_APIKEY}")
    j_resp = resp.json()
    book_list = []
    for i in range(5):
        try:
            result = j_resp["items"][i]
            title = result["volumeInfo"]["title"]
            authors = result["volumeInfo"]["authors"]
            description = result["volumeInfo"]["description"]
            publisher = result["volumeInfo"]["publisher"]
            id = result["id"]
            image = None # TODO
            book_list.append(Book(title, authors, id, description, publisher, image))
        except KeyError:
            return book_list
        except IndexError:
            return book_list
    return book_list

# helper for text_to_book, goes from text -> top 5 isbn results
def text_to_book_list_isbn(text):
    str = '%20'.join(text.split())
    resp = requests.get(f"https://api2.isbndb.com/search/books?text={str}", headers = isbn_header)
    j_resp = resp.json()
    book_list = []
    for i in range(5):
        try:
            result = j_resp["data"][i]
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

# helper for text_to_book, goes from read_text + book_list to an
# excellent or decent choice of book
def book_decider(read_text, book_list):

    if book_list == []:
        return (None, None)

    decent_book = None

    for book in book_list:
        title_word_list = re.sub(r'[^A-Za-z0-9 ]+', '', book.title).lower().split()

        # summary + study guide cases
        if "summary" in title_word_list:
            if not in_string_ish("summary", read_text):
                print(f"{book.title} rejected with 'summary' in title.")
                continue
        if "study" in title_word_list and "guide" in title_word_list:
            if not in_string_ish("study", read_text) or not in_string_ish("guide", read_text):
                print(f"{book.title} rejected with 'study guide' in title.")
                continue

        split_read_text = read_text.lower().split()
        split_read_text.sort()
        title_word_list.sort()

        # read text == book title case
        if title_word_list == split_read_text:
            print(f"{book.title} by {book.authors[0]} accepted with perfect title")
            return (book, None)

        total_words = len(split_read_text)
        fillers_right = 0
        words_in_author = 0
        words_in_title = 0
        words_in_publisher = 0

        # counting
        for word in split_read_text:
            in_author = False
            filler = False

            # fillers
            if word == "the" or word == "an":
                filler = True

            # author
            for author in book.authors:
                if in_string_ish(word, author):
                    if filler:
                        fillers_right += 1
                    else:
                        words_in_author += 1
                    in_author = True
                    break

            # title + publisher
            if not in_author:
                if in_string_ish(word, book.title):
                    if filler:
                        fillers_right += 1
                    else:
                        words_in_title += 1
                elif in_string_ish(word, book.publisher):
                    if filler:
                        fillers_right += 1
                    else:
                        words_in_publisher += 1

        accuracy = (fillers_right + words_in_title + words_in_author + words_in_publisher)/total_words
        print(f"Accuracy is {accuracy*100}%")
        print(f"total words = {total_words}")
        print(f"title = {words_in_title}")
        print(f"author = {words_in_author}")
        print(f"publisher = {words_in_publisher}")
        print(f"fillers = {fillers_right}")

        # An excellent choice requires an accuracy > .70 and words_in_title > 1
        # as well as words_in_title > 1 or words_in_publisher > 1.
        # A decent choice requires accuracy > .50 and words_in_title > 1

        if (accuracy > .7) and (words_in_title >= 1) and (words_in_author >= 1 or words_in_publisher >= 1):
            print(f"{book.title} by {book.authors} accepted")
            return (book, None)
        elif (decent_book == None) and (accuracy > .5) and (words_in_title >= 1):
            decent_book = book
            print(f"{book.title} by {book.authors} is a decent option")
        else:
            print(f"{book.title} by {book.authors} rejected")

    return (None, decent_book)

# key function, goes from text from image_reader to a book
def text_to_book(book_text_pair):
    second_choice = None
    for book_text in book_text_pair:
        if len(book_text.split()) < 2:
            print("read_text length less than 2... rejecting")
            return None
        books = text_to_book_list_isbn(book_text)
        great_option, decent_option = book_decider(book_text, books)
        if great_option != None:
            print("isbn got it...")
            return great_option # done by isbn
        else:
            if second_choice == None:
                second_choice = decent_option
            print("trying google api...")
            books = text_to_book_list_google(book_text)
            great_option, decent_option = book_decider(book_text, books)
            if great_option != None:
                print("google got it...")
                return great_option # done by google
            elif second_choice == None:
                second_choice = decent_option
    return second_choice
