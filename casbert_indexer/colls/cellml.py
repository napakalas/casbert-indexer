from .pmrcollection import PmrCollection
import os

class Cellmls(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
        self.statusC = {'deprecated': 0, 'current': 1, 'validating': 2, 'invalid': 3}
        self.id2Url = {v['id']:k for k, v in self.data.items()}
        self.local2Url = {v['workingDir']+'/'+v['cellml']:k for k, v in self.data.items()}

    def getUrl(self, localPath=None, id=None):
        if localPath != None:
            if localPath in self.local2Url:
                return self.local2Url[localPath]
        elif id != None:
            if id in self.id2Url:
                return self.id2Url[id]
        return None

    def getObjData(self, url=None, localPath=None, id=None, items=[], isCopy=False):
        if localPath != None:
            url = self.getUrl(localPath=localPath)
        elif id != None:
            url = self.getUrl(id=id)
        if url != None:
            if url in self.data:
                return PmrCollection.getObjData(self, id=url, items=items, isCopy=isCopy)
        return {}

    def getWorkspace(self, url=None, localPath=None, id=None):
        objData = self.getObjData(url=url, localPath=localPath, id=id)
        return objData['workspace']

    def getObjLeaves(self, url=None, localPath=None, id=None):
        objData = self.getObjData(url=url, localPath=localPath, id=id)
        leaves = []
        if 'rdfLeaves' in objData:
            for leaf in objData['rdfLeaves']:
                if not leaf.startswith('file:'):
                    leaves += [leaf]
        return leaves

    def getImages(self, url=None, localPath=None, id=None):
        objData = self.getObjData(url=url, localPath=localPath, id=id)
        return objData['images'] if 'images' in objData else []

    def getSedmls(self, url=None, localPath=None, id=None):
        objData = self.getObjData(url=url, localPath=localPath, id=id)
        return objData['sedml'] if 'sedml' in objData else []

    def getCaption(self, url=None, localPath=None, id=None):
        objData = self.getObjData(url=url, localPath=localPath, id=id)
        return objData['caption'] if 'caption' in objData else []

    def getId(self, url=None, localPath=None):
        if localPath != None:
            url = self.getUrl(localPath=localPath)
        if url != None:
            if url in self.data:
                return self.data[url]['id']
        return None

    def getPath(self, url=None, id=None):
        if id != None:
            url = self.getUrl(id=id)
        if url != None:
            if url in self.data:
                return os.path.join(self.data[url]['workingDir'], self.data[url]['cellml'])
        return None
