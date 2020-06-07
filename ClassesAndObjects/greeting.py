# this is example of both method and constructor (method by means of calling the greeting, and constructor by defining name and self within the class Person; and not separate line)

class Person:
    def __init__(self, name):
        self.name = name
    def greeting(self):
        # Should return "hi, my name is " followed by the name of the Person.
        print("hi, my name is {}".format(self.name))

# Create a new instance with a name of your choice
some_person = Person("Tom") 
# Call the greeting method
print(some_person.greeting())
