class Animal:
    name = ""
    category = ""

    def __init__(self, name):
        self.name = name

    def set_category(self, category):
        self.category = category


class Zoo:
    def __init__(self):
        self.current_animals = {}

    def add_animal(self, animal):
        self.current_animals[animal.name] = animal.category

    def total_of_category(self, category):
        result = 0
        for animal in self.current_animals.values():
            if animal == category:
                result += 1
        return result


Turtle = Animal('Turtle')
Snake = Animal('Snake')
Turtle.set_category('Reptile')
Snake.set_category('Reptile')
zoo = Zoo()
print(Snake.category)
zoo.add_animal(Turtle)
zoo.add_animal(Snake)
print(zoo.total_of_category('Reptile'))
