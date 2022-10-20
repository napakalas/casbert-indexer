from .pmrcollection import PmrCollection

class Sedmls(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
        self.id2Url = {v['id']:k for k, v in self.data.items()}
        self.statusC = {'deprecated': 0, 'current': 1, 'validating': 2, 'invalid': 3}

    def getUrl(self, id):
        return self.id2Url[id] if id in self.id2Url else None

    def getVariables(self, id, plot=None):
        url = self.getUrl(id)
        variables = set()
        if plot == None:
            return self.data[url]['variables']
        else:
            series = self.data[url]['outputs'][plot]
            for seri in series:
                variables.add(seri['x'])
                variables.add(seri['y'])
        return list(variables)

    def getWorkspace(self, id):
        url = self.getUrl(id)
        return self.data[url]['workspace']

    def getCellmlId(self, id):
        url = self.getUrl(id)
        return self.data[url]['models']['model']
