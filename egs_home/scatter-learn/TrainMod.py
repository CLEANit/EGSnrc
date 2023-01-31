import numpy as np
import random
from sklearn import *
import sklearn as sk

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

def TestTrainInds(N, tr_frac=8):
    # Training fraction
    ti = (tr_frac-1)*N//tr_frac
    tr_inds = np.arange(N)
    random.shuffle(tr_inds) #... this should be shuffled anyways.
    tri = tr_inds[:ti]
    tei = tr_inds[ti+1:]
    return tri, tei


def SplitTestTrain(ssc, N):
    tri, tei  = TestTrainInds(N)

    ssr = ssc[tri]
    sse = ssc[tei]

    f = ssc.shape[-1]
    input_data_train = ssr.reshape((-1,f*f))
    input_data_test = sse.reshape((-1,f*f))
    return input_data_train, input_data_test, tri, tei


#def TrainModel(input_data_train, lr, input_data_test, le, ccp_alpha):
def TrainModel(input_data_train, lr, ccp_alpha):
    from sklearn import tree,neural_network,ensemble,svm
    #base_mod = sk.tree.DecisionTreeRegressor(ccp_alpha=0.02254,
    #                                         random_state=random_seed)
    
    #agg_mod = sk.tree.DecisionTreeRegressor(random_state=random_seed)
    #agg_mod = sk.svm.LinearSVR()
    #agg_mod = ensemble.RandomForestRegressor()
    #
    #layer_size = ntree + f*f
    #agg_mod= sk.neural_network.MLPRegressor(hidden_layer_sizes=(layer_size,
    #                                                            layer_size,
    #                                                            1),
    #                                        activation='relu',
    #                                        learning_rate='adaptive',
    #                                        solver='adam',
    #                                        n_iter_no_change=20
    #)
    
    #from StackedBootstrapRegressor import StackedBootstrapRegressor
    #mod = StackedBootstrapRegressor(base_mod=base_mod, agg_mod=agg_mod)

    mod = ensemble.RandomForestRegressor(random_state=random_seed,
                                         ccp_alpha=ccp_alpha)

    #mod = sk.tree.DecisionTreeRegressor(ccp_alpha=0.02254,
    #                                         random_state=random_seed)
    #mod = ensemble.GradientBoostingRegressor(random_state=random_seed,ccp_alpha=ccp_alpha)
    
    mod.fit(input_data_train,
                          lr)
    return mod

    #r2 = mod.score(input_data_test, le)
    #print(f"r2: {r2}, ccp: {ccp_alpha}")
    #return mod, r2


class HyperParameterOptimizer():
    def __init__(self):
        self.results = []
        self.val_ind = 0
        self.score_ind = 1

    def AppendResults(self, val, score):
        results_i = [0.0, 0.0]
        results_i[self.val_ind] = val
        results_i[self.score_ind] = score

        self.results.append(results_i)

   
    def OptimalValue(self):
        results_test = np.array(self.results)
        optimal_ind = np.argmax(results_test[:, self.score_ind])
        print(f"Optimal val: {self.results[optimal_ind][self.val_ind]}, ",
              f"yeilds score: {self.results[optimal_ind][self.score_ind]}")

        return self.results[optimal_ind]

    def TestVal(self,  input_data_train, lr, input_data_test, le, ccp_alpha):
        mod, r2 = TrainModel(input_data_train, lr, input_data_test, le, ccp_alpha)
        self.AppendResults(ccp_alpha, r2)
        return mod, r2

        
