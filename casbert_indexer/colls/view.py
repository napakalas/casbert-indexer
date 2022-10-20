from .pmrcollection import PmrCollection

class Views(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
