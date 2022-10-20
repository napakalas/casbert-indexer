from ..general import getUrlFromPmr, getJsonFromPmr
from ..general import PMR_SERVER, RESOURCE_DIR, RS_VIEW
from ..colls.category import Categories
from .view import Views

class Categories(Categories):
    def __init__(self, *paths):
        super().__init__(*paths)

    # get list of categories in the PMR
    def getListCategories(self, fromServer=False):
        if fromServer:
            return getUrlFromPmr(PMR_SERVER)[4:-3]
        else:
            return list(self.data.keys())

    # update local categories
    def update(self):
        views = Views(RESOURCE_DIR, RS_VIEW)
        # get of update categories
        listCategories = self.getListCategories(fromServer=True)
        print('Extracting %d categories'%len(listCategories))
        print('updating %d categories ...'%len(listCategories))
        for counter, category in enumerate(listCategories):
            print(counter, end=' ')
            listViews = getJsonFromPmr(category)
            if 'links' in listViews:
                txtCat = category[category.rfind('/') + 1:]
                self.data[txtCat] = []  # listViews['links']
                for view in listViews['links']:
                    txtView = view['href'][len(PMR_SERVER):]
                    prompt = view['prompt']
                    rel = view['rel']
                    self.data[txtCat] += [txtView]
                    views.add(txtView, prompt, rel, txtCat)
        # save list of categories file
        self.dumpJson()
        # save list of views file
        views.dumpJson()

    def extract(self):
        self.dumpJson()
