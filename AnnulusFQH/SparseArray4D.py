class SparseArray4D:
    def __init__(self):
        self.myDict = dict()

    def getValue(self, x, y, z, w):
        if self.myDict.__contains__((x, y, z, w)):
            return self.myDict[(x, y, z, w)]
        return 0

    def setValue(self, x, y, z, w, val):
        self.myDict[(x, y, z, w)] = val

    def getSize(self):
        return len(self.myDict)