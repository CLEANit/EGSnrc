import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, npad, obj_len, target_len):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, npad+1, stride=1)
        self.c2 = nn.Conv2d(16, 32, npad+1, stride=1)
        self.l1 = nn.Linear(32*64, 4)
        self.l2 = nn.Linear(4,1)
        self.conv= nn.Sequential(
                self.c1, #nn.Conv2d(3, 6, 5,stride=1,padding=16)
                nn.Tanh(),
                #nn.Softmax(),
                self.c2, #nn.Conv2d(6,16,5,stride=1),
        #        nn.Tanh()
        )

        self.npad = npad
        self.obj_len = obj_len


    def forward(self,x):
        x = self.conv(x)
        return x

class CNN_window(nn.Module):
    def __init__(self, npad, obj_len, target_len):
        super(CNN_window, self).__init__()
        self.c1 = nn.Conv2d(1, 32, npad+1, stride=1)
        self.c2 = nn.Conv2d(32, 1, npad+1, stride=1)
        self.conv= nn.Sequential(
                self.c1,
                nn.Tanh(),
                self.c2)
        #, #nn.Conv2d(6,16,5,stride=1),
                #nn.Tanh()
        #)

        self.npad = npad
        self.obj_len = obj_len
        self.target_len = target_len


        #self.l1 = nn.Linear(32*64, 4)
        #self.l2 = nn.Linear(4,1)
        #self.lin = nn.Sequential(
        #            self.l1,
        #            nn.Tanh(),
        #            self.l2)

    def forward(self,x):
        x = self.conv(x)
        #x = x.reshape((-1, 1, 32*64))
        #x = self.lin(x)
        return x

