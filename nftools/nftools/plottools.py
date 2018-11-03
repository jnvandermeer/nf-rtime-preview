class LimitedList(list):

    # Read-only
    @property
    def maxLen(self):
        return self._maxLen

    def __init__(self, *args, **kwargs):
        self._maxLen = kwargs.pop("maxLen")
        list.__init__(self, *args, **kwargs)

    def _truncate(self):
        """Called by various methods to reinforce the maximum length."""
        dif = len(self)-self._maxLen
        if dif > 0:
            self[:dif]=[]

    def append(self, x):
        list.append(self, x)
        self._truncate()

    def insert(self, *args):
        list.insert(self, *args)
        self._truncate()

    def extend(self, x):
        list.extend(self, x)
        self._truncate()

    def __setitem__(self, *args):
        list.__setitem__(self, *args)
        self._truncate()

    def __setslice__(self, *args):
        list.__setslice__(self, *args)
        self._truncate()