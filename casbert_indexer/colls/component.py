from .pmrcollection import PmrCollection

class Components(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)

    def getT2Id(self, ids=None):
        if isinstance(ids,list):
            return {self.data[id]['name']:id for id in ids}
        return {}
        # return {v['name']:k for k,v in self.data.items()}

    def getVariables(self, compId):
        if 'variables' in self.data[compId]:
            return self.data[compId]['variables']
        else:
            # some components such as parent components may not have variables
            # but they have children components which may have variable
            return []

    def getName(self,compId):
        return self.data[compId]['name']

    def getCellml(self, id):
        return self.data[id]['cellml']
