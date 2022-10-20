from ..general import getJsonFromPmr, getUrlFromPmr, PMR_SERVER
from ..colls.exposure import Exposures

class Exposures(Exposures):
    def __init__(self, workspaces, *paths):
        super().__init__(workspaces, *paths)

    def getListExposures(self, fromServer=False):
        if fromServer:
            listExposure = getUrlFromPmr(PMR_SERVER + 'exposure')
            tmp = [url[url.find('.org/') + 5:] for url in listExposure]
            return tmp
        else:
            return list(self.data.keys())

    # update local exposures,
    def update(self):
        # get of update exposures
        listExposures = self.getListExposures(fromServer=True)
        print('updating %d exposures ...' % len(listExposures))
        for counter, url in enumerate(listExposures):
            if counter % 20 == 0: print(counter,end=' ')
            try:
                fullUrl = PMR_SERVER + url
                c = getJsonFromPmr(fullUrl)
                workspaceData = {}
                # extract href
                workspaceData['href'] = c['href']
                # extract items
                for item in c['items']:
                    for k,v in item.items():
                        if k=='data':
                            for datum in v:
                                workspaceData[datum['name']] = datum['value']
                # extract links
                for link in c['links']:
                    link['href'] = link['href'][link['href'].find('.org/') + 5:]
                workspaceData['workspace'] = c['links'][-1]['href'] if len(c['links']) > 0 else ''
                workspaceData['views'] = c['links'][:-1] if len(c['links']) > 1 else []
                # extract version
                workspaceData['version'] = c['version']
                workspaceData['workingDir'] = self.wksData[workspaceData['workspace']]['workingDir'] if workspaceData['workspace'] in self.wksData else ''
                workspaceData['status'] = self.statusC['current']
                self.data[url] = workspaceData
            except Exception as e:
                print('\n',e,url,'\n')
        # update status of deprecated exposure to 0
        for exposure in self.data:
            self.data[exposure]['status'] = self.statusC['deprecated'] if exposure not in listExposures else self.statusC['current']
        # sychronised exposures and self.wksData
        self.__syncExpAndWks()

    # update local date, first update workspace, then exposure, then syncrhronise workspace and exposure
    def __syncExpAndWks(self):
        for exposureUri in self.data:
            workspaceUri = self.data[exposureUri]['workspace']
            if workspaceUri !='':
                commitId = self.data[exposureUri]['commit_id'] if 'commit_id' in self.data[exposureUri] else ''
                if workspaceUri in self.wksData:
                    if 'exposures' not in self.wksData[workspaceUri]:
                        self.wksData[workspaceUri]['exposures'] = {}
                    self.wksData[workspaceUri]['exposures'][exposureUri] = commitId

        # save list of workspaces file
        self.workspaces.dumpJson()
        # save list to exposures file
        self.dumpJson()

    def extract(self):
        self.dumpJson()
