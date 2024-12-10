
import numpy as np
import pandas as pd

from universal.algos.SubSet import readPKL


class BalanceGen():

    def __init__(self, history, filepath, tau=None):
        self.history = history
        self.filepath = filepath
        self.tau = tau

    def init_weights(self, m):
        return np.ones(m) / m

    def update(self):
        pass

    def getEntireBalance(self, b):
        """number of b is not equal to dataset,add stock out of b"""
        df = readPKL(self.filepath)
        nstocks = df.shape[1]
        balance = np.zeros(nstocks)
        itemlists = list(df.iloc[:0])
        bItem = list(b.index)
        for i in range(len(bItem)):
            for j in range(len(itemlists)):
                if bItem[i] == itemlists[j]:
                    balance[j] = b[i]
        balance = pd.Series(balance, index=itemlists)
        return balance
