from sklearn import *
import sklearn as sk
import numpy as np

class StackedBaggingModel():
    def __init__(self, 
                 base_mod=sk.tree.DecisionTreeRegressor(),
                 base_ntree=32,
                 base_bootstrap=True,
                 base_bootstrap_features=True,
                 agg_mod=sk.tree.DecisionTreeRegressor()
                ):


        self.chorus = sk.ensemble._bagging.BaggingEnsemble(  
                                        base_estimator=base_mod,
                                        n_estimators=base_ntree,
                                        bootstrap=base_bootstrap,
                                        bootstrap_features=base_bootstrap_features,
                                        n_jobs=-1
        )

        self.agg_mod = agg_mod

    def _PrepareEnsembleOutput(self, dtr):
       chorus_output = self.chorus.predict(dtr)
       chorus_output = np.array(chorus_output)
       chorus_output = chorus_output.T
       chorus_output = np.hstack((chorus_output, dtr))
       return chorus_output

    def fit(self, dtr, ltr):
       self.chorus.fit(dtr, ltr)
       chorus_output = self._PrepareEnsembleOutput(dtr)
       self.agg_mod.fit(chorus_output, ltr)

    def predict(self, data):
        #chorus_output = self.mod.predict(data)
        chorus_output = self._PrepareEnsembleOutput(data)
        prediction = self.agg_mod.predict(chorus_output)
        return prediction

    def score(self, data, labels):
        chorus_output = self._PrepareEnsembleOutput(data)
        r2 = self.agg_mod.score(chorus_output, labels)
        return r2


        


mod = StackedBaggingModel()
