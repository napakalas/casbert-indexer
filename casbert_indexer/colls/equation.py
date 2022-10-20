from .pmrcollection import PmrCollection

class Maths(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)

    def getText(self, id):
        return self.data[id]
