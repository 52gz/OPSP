

import heapq
import numpy as np
import pandas as pd
from universal import tools
from universal.algos.BalanceGen import BalanceGen

class Ek_b(BalanceGen):

    def __init__(self, history, filepath, tau, epsilon, choice):
        super().__init__(history, filepath, tau)
        self.epsilon = epsilon

        self.choice = choice

    def init_weights(self, m):
        return np.ones(m) / m

    def calculate_last_b(self, history):
        """
        calculate the last b
        :param history:
        """
        last_b = history.iloc[-1] / history.iloc[-2]
        last_b = tools.simplex_proj(last_b)
        return last_b

    def calculate_max_p(self, last_window_history):
        """
        calculate max price of nearly history
        :param history: price sequence of subset
        :param tau: time windows
        :return:
        """

        return last_window_history.max() / last_window_history.iloc[-1]

    def calculate_max_p_T(self, max_p, asset_amount):
        unit_vector = np.ones(max_p.shape[0])
        max_p_t = max_p - (np.dot(unit_vector, max_p) / asset_amount) * unit_vector
        return max_p_t

    def update_new_b(self, max_p, max_p_t, last_b_k):
        condition1 = np.dot(max_p, max_p_t)
        if condition1 == 0:
            return last_b_k

        condition2 = last_b_k + (self.epsilon * max_p_t) / (condition1 ** 0.5)

        if condition2.min() < 0:
            k = map(list(max_p).index, heapq.nlargest(1, list(max_p)))
            k = list(k)
            k = k[0]
            max_p = np.zeros(max_p.shape[0])
            max_p[k] = 1
            last_b_k = max_p
        else:
            last_b_k = condition2

        last_b_k = tools.simplex_proj(last_b_k)
        return last_b_k

    def update(self):
        last_window_history = self.history[-self.tau-1:]
        max_p = self.calculate_max_p(last_window_history)
        max_p_t = self.calculate_max_p_T(max_p, last_window_history.shape[1])
        #
        last_b = self.calculate_last_b(self.history)
        b = self.update_new_b(max_p, max_p_t, last_b)
        b = pd.Series(data=b, index=max_p_t.index)
        b = self.getEntireBalance(b)  # t the number of stocks equal to dataset
        return b
