from Filters import BaseFilter

class Concatenate(BaseFilter):
    '''Concatenates two filters.'''

    def __init__(self, filter1, filter2):
        """Constructor."""
        self._filter1 = filter1
        self._filter2 = filter2

    def filter(self, input):
        return self._filter2.filter(self._filter1.filter(input))
