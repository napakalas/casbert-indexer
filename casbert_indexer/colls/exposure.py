from .pmrcollection import PmrCollection

class Exposures(PmrCollection):
    def __init__(self, workspaces, *paths):
        super().__init__(*paths)
        self.workspaces = workspaces
        self.wksData = workspaces.getJson()['data']
        self.statusC = {'deprecated': 0, 'current': 1, 'validating': 2}
