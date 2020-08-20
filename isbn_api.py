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

# Returns true if edit distance between s1 and s2 is one, else false
def isEditDistanceOne(s1, s2):
    m = len(s1)
    n = len(s2)
    if abs(m - n) > 1:
        return False
    count, i, j = (0, 0, 0)
    while i < m and j < n:
        if s1[i] != s2[j]: # If current characters don't match
            if count == 1:
                return False
            if m > n: # If length of one string is more, then remove
                i+=1
            elif m < n:
                j+=1
            else:    # If lengths of both strings is same, inc both
                i+=1
                j+=1
            count+=1 # Increment count of edits
        else:    # if current characters match
            i+=1
            j+=1
    if i < m or j < n: # if last character is extra in any string
        count+=1
    return count == 1

# helper for choose_books, checks if word is "in" a sentence
def word_in_string(word, sentence):
    string_list = re.sub(r'[^A-Za-z0-9 ]+', '', sentence).lower().split()
    for str in string_list:
        if word == str or (len(word) > 2 and isEditDistanceOne(word, str)):
            return True
    return False

# helper for text_to_book, goes from read_text + book_list to an
# excellent or decent choice of book
def choose_books(read_text, book_list):
    if book_list == []:
        return (None, None)

    decent_book = None

    if len(read_text.split()) < 2:
        print("read_text length less than 2... rejecting")
        return (None, None)

    for book in book_list:
        words_in_author = 0
        words_in_booktext = 0
        words_in_publisher = 0
        author_words = []

        split_read_text = read_text.lower().split()
        split_read_text.sort()
        book_title_word_list = re.sub(r'[^A-Za-z0-9 ]+', '', book.title).lower().split()
        book_title_word_list.sort()

        if (book_title_word_list == split_read_text) and (len(split_read_text) == len(book_title_word_list)):
            print(f"{book.title} by {book.authors[0]} accepted with perfect title")
            return (book, None)

        # for each word read...
        for word in split_read_text:
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
            if not word in author_words:
                if word_in_string(word, book.title):
                    words_in_booktext += 1
                elif word_in_string(word, book.publisher):
                    words_in_publisher += 1
        if words_in_booktext >= 2 and (words_in_booktext - words_in_author) != 0:
            if words_in_author >= 1 or words_in_publisher >= 1:
                try:
                    print(f"{book.title} by {book.authors[0]} accepted with {author_words} in author")
                except IndexError:
                    print(f"{book.title} by {book.authors} accepted with {author_words} in author")
                return (book, None)
            elif decent_book == None:
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

        else:
            try:
                print(f"{book.title} by {book.authors[0]} rejected")
            except IndexError:
                print(f"{book.title} by {book.authors} rejected")
    return (None, decent_book)

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

# key function, goes from text from image_reader to a book
def text_to_book(book_text_pair):
    second_choice = None
    for book_text in book_text_pair:
        if len(book_text) < 2:
            return None
        books = text_to_book_list_isbn(book_text)
        great_option, decent_option = choose_books(book_text, books)
        if great_option != None:
            print("isbn got it...")
            return great_option # done by isbn
        else:
            if second_choice == None:
                second_choice = decent_option
            print("trying google api...")
            books = text_to_book_list_google(book_text)
            great_option, decent_option = choose_books(book_text, books)
            if great_option != None:
                print("google got it...")
                return great_option # done by google
            elif second_choice == None:
                second_choice = decent_option
    return second_choice
