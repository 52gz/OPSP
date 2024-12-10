from universal.algo import Algo
import pandas as pd

from universal.algos.Ek_b import Ek_b
from universal.algos.WssSubSet import WssSubSet
from universal.result import AlgoResult, ListResult


class OSPS(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, datasetname,
                 tau=5,
                 window=5, eps=100,
                 percentage=0.5,
                 ):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OSPS, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        self.histLen = 0  # yjf.
        self.datasetname = datasetname
        self.percentage = percentage
        self.filepath = "./data/" + datasetname + ".pkl"
        self.history = None
        self.filepath1 = "./returnSave/" + datasetname + ".csv"
        self.returnFile = pd.read_csv(self.filepath1)
        self.tau = tau

        self.choice = ''

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", self.histLen, self.datasetname)

        BAH_return = self.returnFile.iloc[self.histLen - 2]["BAH"]
        OLMAR_return = self.returnFile.iloc[self.histLen - 2]["OLMAR"]
        if OLMAR_return >= BAH_return:
            # follow the loser
            # remove the high return subset
            self.choice = 'loser'
            tool = WssSubSet(history, dataset=self.datasetname, percentage=self.percentage, index=1)
        else:
            # follow the winner
            # remove the low return subset
            self.choice = 'winner'
            tool = WssSubSet(history, dataset=self.datasetname, percentage=self.percentage, index=0)

        # cut data
        history = tool.cutDataset(ndays=self.histLen)
        self.history = history

        # update portfolio
        tool = Ek_b(history=self.history, filepath=self.filepath, tau=self.tau, epsilon=self.eps, choice=self.choice)

        b = tool.update()

        # progress
        print('\r', "[" + str(self.histLen) + "]", end=" ", flush=True)
        return b


if __name__ == '__main__':

    datasetList = ['djia', 'hs300', 'msci', 'sp500', 'tse', 'nyse_n']

    for dlist in datasetList:
        path = '../data/' + dlist + '.pkl'
        df_original = pd.read_pickle(path)
        t = OSPS(dlist, percentage=0.3)
        df = t._convert_prices(df_original, 'raw')
        B = t.weights(df)
        Return = AlgoResult(t._convert_prices(df_original, 'ratio'), B)
        res = ListResult([Return], ['OSPS']).to_dataframe()
        last_return = res.iloc[-1].values[0]
        print(dlist + ": last_return = ", last_return)
        print("====================" + dlist + " done=======================")
