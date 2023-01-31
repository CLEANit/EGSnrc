from sklearn import *
import sklearn as sk
import numpy as np

class ResidualModel():
    def __init__(self, 
                 base_mod,
                 residual_mod=None
                ):

        self.mod1 = base_mod
        
        if residual_mod is None:
            self.residual_mod = base_mod
        else:
            self.residual_mod = residual_mod


    def score(self, X, y):
        return sk.base.RegressorMixin.score(self, X, y)


    def fit(self, dtr1, ltr, dtr_res=None):
        self.mod1.fit(dtr1, ltr)
        residuals = ltr - self.mod1.predict(dtr1) 

        if dtr_res is None:
            self.residual_mod.fit(dtr1, residuals)

        else:
            self.residual_mod.fit(dtr_res, residuals)


    def predict(self, data1, data_res=None):
        pred1 = self.mod1.predict(data1)

        if data_res is None:
        # mod.predict(None) -> ValueError
            pred_res = self.residual_mod.predict(data1)

        else:
            pred_res = self.residual_mod.predict(data_res)

        prediction = pred1 + pred_res
        return prediction

