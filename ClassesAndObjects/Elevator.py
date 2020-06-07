class Elevator:
    def __init__(self, bottom, top, current):
        """Initializes the Elevator instance."""
        self.bottom = bottom
        self.top = top
        self.current = current
    def up(self):
        """Makes the elevator go up one floor."""
        self.current = self.current + 1 if self.current < self.top else self.current
    def down(self):
        """Makes the elevator go down one floor."""
        self.current = self.current - 1 if self.current > self.bottom else self.current
    def go_to(self, floor):
        """Makes the elevator go to the specific floor."""
        self.current = floor
    def __str__(self):
        return '"Current floor: {}"'.format(self.current)

elevator = Elevator(-1, 10, 0)
