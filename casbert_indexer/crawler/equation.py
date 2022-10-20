from ..colls.equation import Maths

class Maths(Maths):
    def __init__(self, *paths):
        super().__init__(*paths)

    def add(self, text):
        mathId = self.getNewId()
        self.data[mathId] = text
        return mathId
