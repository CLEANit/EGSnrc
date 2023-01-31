from sklearn import *
import sklearn as sk
import numpy as np

class StackedBootstrapRegressor():
    def __init__(self, 
                 base_mod=None,
                 n_estimators=100,
                 base_bootstrap=True,
                 base_bootstrap_features=True,
                 agg_mod=None
                ):

        #super().__init__(
        #        base_mod=base_mod,
        #        n_estimators = n_estimators,
        #        base_bootstrap = base_bootstrap,
        #        base_bootstrap_features = base_bootstrap_features,
        #        agg_mod = agg_mod
        #)

        self.chorus = sk.ensemble._bagging.BaggingEnsembleRegressor(  
                                        base_estimator = base_mod,
                                        n_estimators = n_estimators,
                                        bootstrap = base_bootstrap,
                                        bootstrap_features = base_bootstrap_features,
                                        n_jobs=-1
        )

        self.agg_mod = agg_mod

    def _PrepareEnsemblePredict(self, dtr):
       chorus_output = self.chorus.predict(dtr)
       chorus_output = np.array(chorus_output)
       #chorus_output[1:,:] /= chorus_output[0,:]
       chorus_output = chorus_output.T
       #print(f"chorus_output.shape: {chorus_output.shape}")
       #print(f"dtr.shape: {dtr.shape}")
       chorus_output = np.hstack((chorus_output, dtr))
       return chorus_output

    #def _PrepareEnsemblePredict_Proba(self, dtr):
    #   chorus_output = self.chorus.predict_proba(dtr)
    #   chorus_output = np.array(chorus_output)
    #   chorus_output = chorus_output.T
    #   #print(f"chorus_output.shape: {chorus_output.shape}")
    #   #print(f"dtr.shape: {dtr.shape}")
    #   #chorus_output = np.hstack((chorus_output, dtr))
    #   return chorus_output

    def fit(self, dtr, ltr):
       self.chorus.fit(dtr, ltr)
       chorus_output = self._PrepareEnsemblePredict(dtr)
       self.agg_mod.fit(chorus_output, ltr)

    def predict(self, data):
        #chorus_output = self.mod.predict(data)
        chorus_output = self._PrepareEnsemblePredict(data)
        prediction = self.agg_mod.predict(chorus_output)
        return prediction

    #def predict_proba(self, data):
    #    #chorus_output = self.mod.predict(data)

    # # Logical thing would be to start as normal..
    # #   chorus_output = self._PrepareEnsemblePredict_Proba(data)
    # # And let the aggregating model determine probability.
    # #   prediction = self.agg_mod.predict_proba(chorus_output)

    # # But since the aggregating model is a decision tree - the probability
    # # is always (always?) one. SO, if we proceed in the manner of a Random 
    # # Forest. DEV: CHECK WHAT THE BUILT-IN STACKING ENSEMBLE DOES.
    #    chorus_output = self.chorus.predict(data)
    #    chorus_proba = np.sum(chorus_output, axis=0)
    #    chorus_proba = chorus_proba / data.shape[0]
    #    chorus_proba = np.dstack((1-chorus_proba, chorus_proba))[0]
    #    return chorus_proba 

    def score(self, data, labels):
        chorus_output = self._PrepareEnsemblePredict(data)
        r2 = self.agg_mod.score(chorus_output, labels)
        return r2


        


#mod = StackedBaggingModel()
