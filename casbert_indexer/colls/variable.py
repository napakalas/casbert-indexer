from .pmrcollection import PmrCollection

class Variables(PmrCollection):
    def __init__(self, sysMaths, *paths):
        super().__init__(*paths)
        self.sysMaths = sysMaths

    def getT2Id(self, ids=None, short=False):
        if ids != None:
            return {self.getName(id, short):id for id in ids if self.getType(id) != 'rate'}
        return {self.getName(id, short):id for id in self.data if self.getType(id) != 'rate'}

    def getName(self, id, short=False):
        if short:
            return(self.data[id]['shortName'])
        return(self.data[id]['name'])

    def getType(self, id):
        return self.data[id]['type']

    def getInit(self, id):
        return self.data[id]['init']

    def getRate(self, id):
        if self.getType(id) == 'state':
            return self.data[id]['rate']
        return None

    def getMaths(self, id):
        varMaths = {}
        if 'math' in self.data[id]:
            for mathId in self.data[id]['math']:
                varMaths[mathId] = self.sysMaths.getText(mathId)
        return varMaths

    def getDependents(self, id, varDep={}):
        if 'dependent' in self.data[id]:
            if len(self.data[id]['dependent']) > 0:
                for varIdDep, varNameDep in self.data[id]['dependent'].items():
                    if varIdDep not in varDep:
                        varDep[varIdDep]={'name':varNameDep, 'math':self.getMaths(varIdDep), 'type':self.getType(varIdDep), 'init':self.getInit(varIdDep)}
                        if 'dependent' in self.data[varIdDep]:
                            if len(self.data[varIdDep]['dependent']) > 0:
                                self.getDependents(varIdDep, varDep)
                                
    def getUnit(self, id):
        return self.data[id]['unit']

    def getPlots(self, id):
        if 'plot' in self.data[id]:
            return self.data[id]['plot']
        return []

    def getCompId(self, id):
        return self.data[id]['component']
