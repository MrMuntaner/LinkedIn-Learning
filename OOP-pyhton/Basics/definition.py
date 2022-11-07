# Basic class definitions

# TODO: create a basic class

# The init function is one of pythons special functions for working with-
# classes.
# title is an attribute passed in as an argument
class Book:
    def __init__(self, title):
        self.title = title

# TODO: create instances of the class

# This will create a book object
b1 = Book('Brave New World')
b2 = Book('War and Peace')

# TODO: print the class and property

# I can access the value of the property by using dot notation just like any other object
print(b1)
print(b1.title)