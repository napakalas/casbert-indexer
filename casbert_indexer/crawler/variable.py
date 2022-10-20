from ..colls.variable import Variables

class Variables(Variables):
    def __init__(self, sysMaths, *paths):
        super().__init__(sysMaths, *paths)

    def add(self, varId, varValue, name, compId, varType):
        varParts = name.split('/')
        shortName = varParts[-1] if varParts[-1] != 'prime' else varParts[-2]
        self.data[varId] = {'init': varValue, 'name': name, 'shortName': shortName, 'component': compId, 'type': varType}

    def addElement(self, elements):
        variablesT2Id = {}
        for element in elements:
            varName = element.attrib['name']
            if varName in varsT2Id:
                varId = varsT2Id[varName]
                varUnit = element.attrib['units']
                unitId = unitsN2Id[varUnit] if varUnit in unitsN2Id else varUnit
                self.data[varId]['unit'] = unitId
            else:
                varId = mapVars[varName]
            if self.data[varId]['component'] != compId:
                if 'varRefs' not in self.sysComps[compId]:
                    self.sysComps[compId]['varRefs'] = {}
                self.components[compId]['varRefs'][varId] = varName
            self.__setMetaFromElement(element, modelMeta, varId)

    def setUnit(self, id, unitId):
        if unitId != None:
            self.data[id]['unit'] = unitId

    def addDependents(self, id, varDepId, mathVar):
        if 'dependent' not in self.data[id]:
            self.data[id]['dependent'] = {}
        if varDepId in self.data:
            self.data[id]['dependent'][varDepId] = mathVar

    def setMap(self, id, varMapId):
        self.data[id]['map'] = varMapId

    def addMath(self, id, mathText):
        if 'math' in self.data[id]:
            listText = [self.sysMaths.getText(mathId) for mathId in self.data[id]['math']]
            if mathText not in listText:
                mathId = self.sysMaths.add(mathText)
                self.data[id]['math'] += [mathId]
        else:
            mathId = self.sysMaths.add(mathText)
            self.data[id]['math'] = [mathId]

    def addPlot(self, id, plotId):
        if 'plot' in self.data[id]:
            if plotId not in self.data[id]['plot']:
                self.data[id]['plot'] += [plotId]
        else:
            self.data[id]['plot'] = [plotId]

    def setRate(self, id, rate):
        self.data[id]['rate'] = rate
