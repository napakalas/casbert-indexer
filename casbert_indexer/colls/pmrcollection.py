from ..general import loadJson, dumpJson
import copy
import os

class PmrCollection:
    def __init__(self, *paths):
        self.dataDict = loadJson(*paths)
        self.paths = paths
        if len(self.dataDict)==0:
            self.dataDict['data'] = {}
        self.statusC = {}
        self.data = self.dataDict['data']

    def getJson(self):
        return self.dataDict

    def dumpJson(self):
        self.dataDict['status'] = self.statusC
        for k, v in self.dataDict['data'].items():
            if isinstance(v,dict):
                self.dataDict['vars'] = list(v.keys())
            break
        dumpJson(self.dataDict, *self.paths)

    def getStatus(self):
        return self.dataDict['status']

    def getData(self):
        return self.data

    def getObjData(self, id, items=[], isCopy=False):
        if not isCopy:
            return self.data[id]
        if len(items) == 0:
            return copy.deepcopy(self.data[id])
        retObj = {}
        for item in items:
            if item in self.data[id]:
                if item == 'rdfLeaves':
                    retObj[item] = self.getObjLeaves(id)
                else:
                    retObj[item] = self.data[id][item]

        return retObj

    def getObjLeaves(self, id):
        if id in self.data:
            leaves = []
            if 'rdfLeaves' in self.data[id]:
                for leaf in self.data[id]['rdfLeaves']:
                    if not leaf.startswith('file:'):
                        leaves += [leaf]
        return leaves

    def getNewId(self):
        return self.__class__.__name__[:3]+'Id-'+str(len(self.data))

    def addRdf(self, id, rdf, rdfLeaves, cmeta):
        if 'rdf' not in self.data[id]:
            self.data[id]['rdf'] = []
        if 'rdfLeaves' not in self.data[id]:
            self.data[id]['rdfLeaves'] = []
        self.data[id]['rdf'] += rdf
        self.data[id]['rdfLeaves'] += rdfLeaves
        self.data[id]['cmeta'] = cmeta

    def getCMeta(self, id):
        if 'cmeta' in self.data[id]:
            return self.data[id]['cmeta']
        return None
