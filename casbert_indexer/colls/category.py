from .pmrcollection import PmrCollection

class Categories(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
