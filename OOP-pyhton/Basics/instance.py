# Using Instance methods and attributes 

class Book:
    # The 'init' function is called when the instance is
    # created and ready to be initialized
    
    # I will add more attributes for the Book class
    def __init__(self, title, author, pages, price):
        self.title = title
        self.author = author
        self.pages = pages
        self.price = price
    
    # TODO: create another instance method
    def getprice(self):
        return self.price
    
    # TODO: create discount instance method
    # Notice how i have an underscore in front of the attribute name.
    # The reason for this is to give other developers who use this class, a-
    # hint that this class is considered internal to the class and should not be-
    # accessed from outside the class's logic 


# TODO: create some book instances
b1 = Book('War and Peace', 'Leo Tolstoy', 1225, 39.95)
b2 = Book('The Catcher in the Rye', 'JD Salinger', 234, 29.95)

# TODO: print the price of book1
print(b1.getprice())

# TODO: try setting the discount
# I am not limited to creating instance attributes just within the init function.
# I can do it else where in the object as well.