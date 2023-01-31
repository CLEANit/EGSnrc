import numpy as np
import random
from sklearn import *
import sklearn as sk
from StackedBootstrapRegressor import StackedBootstrapRegressor
from random import shuffle
import functools as ft

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

def TestTrainInds(N, tr_frac=2):
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


import torch
def CreateNet(npad, obj_len, target_len):
    from ConvNet import CNN_window
    net = CNN_window(npad, obj_len, target_len)
    #                                        random_state=random_seed)
    return net

def ToTorch(data):
    data = np.array(data, dtype=np.float32)
    data = torch.from_numpy(data)
    data = torch.FloatTensor(data)
    return data


def SquareData(input_data_train, lr): #, input_data_test, le):
    obj_len = np.sqrt(input_data_train.shape[-1])
    obj_len = int(obj_len)
    input_data_train = input_data_train.reshape((-1, 1, obj_len, obj_len))
    #input_data_test = input_data_test.reshape((-1, 1, obj_len, obj_len))
    num_train = input_data_train.shape[0]

    lr = lr.reshape((num_train, -1))
    target_len = lr.shape[-1]
    target_len = np.sqrt(target_len)
    target_len = int(target_len)

    lr = lr.reshape((num_train, 1, target_len, target_len))
    #le = le.reshape((-1, 1, target_len, target_len))
    #return input_data_train, lr #, input_data_test, le, obj_len, target_len
    return input_data_train, lr, obj_len, target_len

def SquareInput(input_data, obj_len, set_size=-1):
    input_data = input_data.reshape((set_size, 1, obj_len, obj_len))
    return input_data

def RangeBatch(batch_size, batch_i):
    batch_start = batch_size * batch_i
    batch_end = batch_size * (batch_i + 1)
    return batch_start, batch_end


def GenBatch(epoch_inds, batch_size, N_batch):
    shuffle(epoch_inds)
    rbf = ft.partial(RangeBatch, batch_size)
    batch_inds = map(rbf, range(N_batch))
    for batch_start, batch_end in batch_inds:
        yield epoch_inds[batch_start: batch_end]

        

def TrainNet(input_data_train, lr, npad=2):
    # prepare input data
    input_data_train, lr, obj_len, target_len = SquareData(input_data_train,
                                                           lr)
    input_data_train = ToTorch(input_data_train)

    # prepare labels
    lr = ToTorch(lr)

    # Create net
    net = CreateNet(npad, obj_len, target_len) 

    learning_rate = 0.025
    opti = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    # Gives negative r2??
    #opti = torch.optim.Adam(net.parameters(), lr=learning_rate)
    crit = torch.nn.MSELoss()
    N_epoch = 2500
    N_batch = 10
    batch_size = len(lr)//N_batch

    net.train()

    epoch_inds = np.arange(len(lr))
    
    for epoch in range(N_epoch):
        batch_inds_gen = GenBatch(epoch_inds, batch_size, N_batch)


        for batch_inds in batch_inds_gen:
            batch_input = input_data_train[batch_inds]
            batch_labels = lr[batch_inds]

            outs = net(batch_input)

            #print(f"{outs.shape}: outs.shape")
            loss = crit(outs, batch_labels)
            loss.backward()
            opti.step()
        print(epoch, loss.item())
        net.zero_grad()

    #net.eval()
    #with torch.no_grad():
    #    outs = net(input_data_test)
    #    loss = crit(outs, le)
    #    print(loss)

    mod = sktorch(net)
    return mod #, input_data_train, input_data_test, lr, le


class sktorch():
    def __init__(self, net):
        net.eval()
        
        self.net = net

    def predict(self, input_data):
        input_data = SquareInput(input_data, self.net.obj_len)
        input_data = ToTorch(input_data)
        outs = self.net(input_data)
        outs = outs.detach().numpy()
        return outs

    def score(self, input_data, labels):
        outs = self.predict(input_data)

        labels = SquareInput(labels, self.net.target_len)


        r2 = self._r2(labels, outs)
        return r2

    def _r2(self, y_true, y_pred):
        
        u = self._ResSumSq(y_true, y_pred)
        v = self._ResSumSq(y_true, y_true.mean())
        r2 = 1 - u/v
        r2 = r2.item()
        return r2

    def _ResSumSq(self, y1, y2):
        #u = y_true - y_pred
        #v = y_true - np.mean(y_true)

        u = y1 - y2
        u = u**2
        u = u.sum()
        return u
    

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

    def TestVal(self,  input_data_train, lr, input_data_test, le):
        mod, r2 = TrainModel(input_data_train, lr, input_data_test, le)
        self.AppendResults(ccp_alpha, r2)
        return mod, r2

        
#batch_input_gen = (input_data_train[ for batch_i in range(N_batch))
#batch_labels_gen = (lr[epoch_inds[batch_size*batch_i:batch_size*(batch_size+1)]] for batch_i in range(N_batch))
