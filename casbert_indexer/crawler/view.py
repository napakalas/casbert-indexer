from ..colls.view import Views

class Views(Views):
    def __init__(self, *paths):
        super().__init__(*paths)

    def add(self, txtView, prompt, rel, txtCat):
        if txtView in self.data:
            if txtCat not in self.data[txtView]['categories']:
                self.data[txtView]['categories'] += [txtCat]
        else:
            self.data[txtView] = {'categories':[txtCat]}
        self.data[txtView]['prompt'] = prompt
        self.data[txtView]['rel'] = rel

    def extract(self):
        self.dumpJson()
