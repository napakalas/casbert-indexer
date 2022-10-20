from .pmrcollection import PmrCollection

class Images(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
        self.statusC = {'unavailable': 0, 'available': 1}

    def getPath(self, id):
        return self.data[id]['path']

    def getTitle(self, id):
        return self.data[id]['title']

    def isAvailable(self, id):
        if self.data[id]['status'] == self.statusC['available']:
            return True
        else:
            return False
