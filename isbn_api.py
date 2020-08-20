import re
import requests
import json

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

# Returns true if edit distance between s1 and s2 is
# one, else false
def isEditDistanceOne(s1, s2):
    m = len(s1)
    n = len(s2)

    # If difference between lengths is more than 1,
    # then strings can't be at one distance
    if abs(m - n) > 1:
        return False

    count = 0    # Count of isEditDistanceOne

    i = 0
    j = 0
    while i < m and j < n:
        # If current characters dont match
        if s1[i] != s2[j]:
            if count == 1:
                return False

            # If length of one string is
            # more, then only possible edit
            # is to remove a character
            if m > n:
                i+=1
            elif m < n:
                j+=1
            else:    # If lengths of both strings is same
                i+=1
                j+=1

            # Increment count of edits
            count+=1

        else:    # if current characters match
            i+=1
            j+=1

    # if last character is extra in any string
    if i < m or j < n:
        count+=1

    return count == 1

def choose_books(read_text, book_list):

    def word_in_string(word, sentence):
        string_list = re.sub(r'[^A-Za-z0-9 ]+', '', sentence).lower().split()
        for str in string_list:
            if word == str or isEditDistanceOne(word, str):
                return True
        return False

    decent_book = None

    if book_list == []:
        return (None, None)

    for book in book_list:
        words_in_author = 0
        words_in_booktext = 0
        words_in_publisher = 0
        author_words = []
        # for each word read...
        for word in read_text.split():
            if word == "the" or word == "an":
                continue
            # check if in author
            for author in book.authors:
                if word_in_string(word, author):
                    words_in_author += 1
                    words_in_booktext += 1
                    in_author = True
                    author_words.append((word, author))
                    break
            if author_words == []:
                if word_in_string(word, book.title):
                    words_in_booktext += 1
                elif word_in_string(word, book.publisher):
                    words_in_publisher += 1
        if words_in_booktext >= 2:
            if words_in_author >= 1 or words_in_publisher >= 1:
                try:
                    print(f"{book.title} by {book.authors[0]} accepted with {author_words} in author")
                except IndexError:
                    print(f"{book.title} by {book.authors} accepted with {author_words} in author")
                return (book, None)
            else:
                if decent_book == None:
                    decent_book = book
                    try:
                        print(f"{book.title} by {book.authors[0]} is a decent option")
                    except IndexError:
                        print(f"{book.title} by {book.authors} is a decent option")
        else:
            try:
                print(f"{book.title} by {book.authors[0]} rejected")
            except IndexError:
                print(f"{book.title} by {book.authors} rejected")

    return (None, decent_book)

def resp_to_book_google(json_resp):
    book_list = []
    for i in range(5):
        try:
            result = json_resp["items"][i]
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

def resp_to_book_isbn(json_resp):
    book_list = []
    for i in range(5):
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

def text_to_book(book_text_pair):
    second_choice = None
    for book_text in book_text_pair:
        str = '%20'.join(book_text.split())
        resp = requests.get(f"https://api2.isbndb.com/search/books?text={str}", headers = isbn_header)
        j_resp = resp.json()
        books = resp_to_book_isbn(j_resp)
        great_option, decent_option = choose_books(book_text, books)
        if great_option != None:
            print("isbn got it...")
            return great_option # done by isbn
        else:
            if second_choice == None:
                second_choice = decent_option
            str = '+'.join(book_text.split())
            resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={str}&key={google_APIKEY}")
            j_resp = resp.json()
            books = resp_to_book_google(j_resp)
            great_option, decent_option = choose_books(book_text, books)
            if great_option != None:
                print("google got it...")
                return great_option # done by google
            elif second_choice == None:
                second_choice = decent_option
    return second_choice
