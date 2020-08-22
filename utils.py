import re

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
def in_string_ish(word, sentence):
    string_list = re.sub(r'[^A-Za-z0-9 ]+', '', sentence).lower().split()
    for str in string_list:
        if word == str or (len(word) > 2 and isEditDistanceOne(word, str)):
            return True
    return False
