from ..colls.image import Images
from ..general import CURRENT_PATH, WORKSPACE_DIR

import os

class Images(Images):
    def __init__(self, *paths):
        super().__init__(*paths)

    def add(self, elements, cellml):
        ids = []
        for figure in elements:
            ns = '{'+figure.nsmap[None] +'}'
            # print(cellml['id'], cellml['workingDir'], cellml['cellml'])
            figId = figure.attrib['id'] if 'id' in figure.attrib else ''
            caption = figure.find('.//' + ns + 'caption')
            figCaption = caption.text if caption != None else ''
            title = figure.find('.//' + ns + 'title')
            figTitle = title.text if title != None else ''
            figFile = figure.find('.//' + ns + 'imagedata').attrib['fileref']
            id = self.getNewId()
            self.data[id] = {'name':figId, 'caption':figCaption,'title':figTitle,'path':figFile,'cellml':cellml['id'], 'status':self.__isAvailable(figFile, cellml)}
            ids += [id]
        return ids

    def __isAvailable(self, figFile, cellml):
        cellmlPath = os.path.join(CURRENT_PATH, WORKSPACE_DIR, cellml['workingDir'],cellml['cellml'])
        imagePath = os.path.join(os.path.dirname(cellmlPath),figFile)
        if os.path.exists(imagePath):
            return self.statusC['available']
        else:
            return self.statusC['unavailable']
