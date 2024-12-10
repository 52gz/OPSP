import math
from scipy.optimize import fminbound
import numpy as np

from universal.algos.SubSet import SubSet


class WssSubSet(SubSet):

    def __init__(self, history, dataset=None, percentage=None, index=None, windowSize=120,
                 sizeRatio=0.3, targetReturn=0.0016, penaltyW=10 ** 5,
                 stepSize=10 ** (-5),
                 targetError=10 ** (-5)):
        """

        :param history:
        :param dataset:
        :param percentage:
        :param index: 1 drop top stock, 0 drop low stock
        """
        super().__init__(history, dataset, percentage, index)
        self.windowSize = windowSize  # T
        self.sizeRatio = sizeRatio  # rho
        self.targetReturn = targetReturn  # h
        self.penaltyW = penaltyW  # beta
        self.stepSize = stepSize  # eta
        self.targetError = targetError  # zeta

    def init_weights(self, m):
        return np.ones(m) / m

    # represents the single-period returns of assets
    def simpleReturn(self, priceSequence, T):
        assetsNum = priceSequence.shape[1]
        days = priceSequence.shape[0]
        result = []
        if days <= T:
            print("The windowsize is too big!")
            return
        for t in range(days - T, days):
            xi = np.zeros(assetsNum)
            for i in range(0, assetsNum):
                temp = (priceSequence.iloc[t, i] - priceSequence.iloc[t - 1, i]) / priceSequence.iloc[t - 1, i]
                xi[i] = temp
            result.append(xi)
        return result

    # alpha = inf{exp(r-h)alpha(r)}
    # alpha(r) = 1/(r+c)^d
    def func_alpha(self, r):
        h = self.targetReturn  # 0.0016
        c = 1.1
        d = 4
        return math.exp(r - h) / math.pow(r + c, d)

    # alpha = inf{exp(r-h)alpha(r)} r in [r1, rk] # [0,1]
    def get_alpha(self, r1, rk):
        return fminbound(self.func_alpha, r1, rk, full_output=1)[1]

    # f(w)
    def funcf_w(self, weigths, simpleReturn, penaltyW, alpha):
        E_xiTw = 0
        E_exp = 0
        for xi in simpleReturn:
            temp = np.dot(xi, weigths)
            E_xiTw += temp
            E_exp += math.exp(0 - temp)

        E_xiTw = E_xiTw / len(simpleReturn)
        E_exp = E_exp / len(simpleReturn)

        return -E_xiTw + penaltyW * (E_exp - alpha)

    # ∇f(w)
    def Nabla_f_w(self, weights, simpleReturn, penaltyW):
        E_xi = np.zeros(len(simpleReturn[0]))
        E_exp_xi = np.zeros(len(simpleReturn[0]))
        for xi in simpleReturn:
            E_xi += xi
            temp = np.dot(xi, weights)
            E_exp_xi += math.exp(0 - temp) * xi

        E_xi = E_xi / len(simpleReturn)
        E_exp_xi = E_exp_xi / len(simpleReturn)

        return -E_xi - penaltyW * E_exp_xi

    #  standard softmax function
    def softmax(self, v):
        newWeights = []
        sum_exp_v = 0
        for j in v:
            sum_exp_v += math.exp(j)

        for i in v:
            temp = math.exp(i) / sum_exp_v
            newWeights.append(temp)
        return newWeights

    # ˆw(k + 1) = w(k) − ηk ∇f(wk)
    def update_weights(self, oldWeights, simpleReturn, learn_ratio):
        Nabla_w_old = self.Nabla_f_w(oldWeights, simpleReturn, self.penaltyW)
        v = oldWeights - learn_ratio * Nabla_w_old
        newWeights = self.softmax(v)
        return newWeights

    # cal the asset ranking weight
    def getWeight(self, windowSize):
        w = self.init_weights(self.history.shape[1])
        xi = self.simpleReturn(self.history, windowSize)
        alpha = self.get_alpha(0, 1)
        beta = self.penaltyW
        k = 0
        w_old = w
        while abs(self.funcf_w(w, xi, beta, alpha) - self.funcf_w(w_old, xi, beta, alpha)) > self.targetError or k < 1:
            w_old = w
            w = self.update_weights(w_old, xi, self.stepSize)
            w = [round(i, 6) for i in w]
            k += 1
        return w

    def getIndex(self, numstocks=None):
        # look back 120 days
        day = self.history.shape[0]
        if day > 120:
            day = 120
        else:
            day = day - 1

        # cal asset-ranking
        b = self.getWeight(day)

        if numstocks == None:
            numstocks = self.calNumstocks()

        # Ascending, return the index before sorting
        sorted_index = np.argsort(b, kind='mergesort')

        lowStockIndex = sorted_index[:numstocks]
        topStockIndex = sorted_index[-numstocks:]
        return lowStockIndex, topStockIndex





