from .crawler.crawler import Crawler
from .indexer.indexer import Indexer
from .searcher.searcher import Searcher
from .tester.tester import Tester

from .general import zipResources


def __extractData():
    import os
    import zipfile
    path = os.path.dirname(os.path.realpath(__file__))
    drive = os.path.join(path, 'resources')
    checkedFile = os.path.join(drive, 'listOfMath.json')
    filePath = os.path.join(drive, 'casbert_index_data.zip')

    if not os.path.exists(checkedFile):
        pz = open(filePath, 'rb')
        packz = zipfile.ZipFile(pz)
        for name in packz.namelist():
            packz.extract(name, path)
        pz.close()


__extractData()
