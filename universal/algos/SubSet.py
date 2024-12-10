import pandas as pd


class SubSet(object):

    def __init__(self, history, dataset=None, percentage=None, index=None):
        self.history = history
        self.dataset = dataset
        self.percentage = percentage
        self.filepath = "./data/" + dataset + ".pkl"
        self.nStocks = history.shape[1]
        self.index = index

    def calNumstocks(self, percentage=None, filepath=None):
        """calculate how many stocks selected"""

        if percentage == None:
            percentage = self.percentage

        if filepath == None:
            filepath = self.filepath

        df = readPKL(filepath)
        numstocks = round(df.shape[1] * percentage)
        return numstocks


    def cutDataset(self, ndays=None):

        # index == 0  -> drop lowStockIndex
        # index == 1  -> drop topStockIndex
        stockIndex = self.getIndex()[self.index]
        df = self.history.copy()
        itemLists = []
        count = 0
        for item in df:
            if count in stockIndex:
                itemLists.append(item)
            count += 1
        df = df.drop(itemLists, axis=1)
        df = df.iloc[:ndays]
        return df

    def getIndex(self, numstocks=None):
        # 1、wss subset
        pass


def readPKL(filepath):
    """read pickle"""
    df = pd.read_pickle(filepath)
    return df