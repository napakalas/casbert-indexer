from .pmrcollection import PmrCollection


class Units(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)

    def getID(self, text):
        if text in self.getT2Id():
            return self.getT2Id()[text]
        else:
            return None

    def getText(self, id):
        if id in self.data:
            return self.data[id]['text']
        else:
            return None

    def getNames(self, id=None, text=None):
        if id in self.data:
            return self.data[id]['names']
        elif text in self.getT2Id():
            return self.data[self.getT2Id()[text]]['names']
        else:
            return []

    def getId2T(self):
        return {k: v['text'] for k, v in self.data.items()}

    def getT2Id(self):
        return {v['text']: k for k, v in self.data.items()}
