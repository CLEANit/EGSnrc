import numpy as np
def slabs(xl,ml,n,i,j):
    xi = xl[i]
    mi = ml[i]
    return bool((xi//mi)%n)

def columns(xl,ml,n,i,j):
    ## i,j switched
    return bool(slabs(xl,ml,n,i,j) == slabs(xl,ml,n,j,i))

def checkers(xl,ml,n,i,j):
    ## j ignored since all three dims are used.
    return bool( bool(slabs(xl,ml,n,0,0) == slabs(xl,ml,n,1,1)) == slabs(xl,ml,n,2,2) )

def null_pat(xl,ml,n,i,j):
    return 0

def ChooseRandomSign():
    direction_ind = np.random.randint(0,2)
    direction_ind *= 2
    direction_ind -= 1
    return direction_ind


#def binary(xi,yi,zi,n):
#    ## For a 2-d (zi not used)
#    #  binary mask, return the 
#    #  material number for a
#    #  given index location.
#    #  (0 - vacuum/atmosphere,
#    #   1 - material)
#
#    ## Extensions
#    #  - add ability for multiple 
#    #    material (for base N)
#    #       > see base_r
#
#    x = bin(n)[2:]
#    x = x.zfill(64)
#
#
#    # Either zero or one
#    direction_ind = ChooseRandomSign()
#    x = x[::direction_ind]
#
#
#    x = x[((xi+4)*8)+(yi+4)]
#    x = int(x)
#    return x#int(x[((xi+4)*8)+(yi+4)])

yindf = lambda x: (x%8-4)
xindf = lambda y: (int((y%64)/8)-4)
zindf = lambda z: int(z/64)

from numpy import base_repr
class base_r():
    def __init__(self, rand_num, base=3):
        try:
            self.rn_base = np.base_repr(rand_num,base=base)
        except:
            self.rn_base = base_repr(rand_num,base=base)

        self.rn_base = self.rn_base.zfill(64)

        # This might improve the distribution..
        #   but it breaks compatibility between random number label and actual
        #   geometery.
        #direction_ind = ChooseRandomSign()
        #self.rn_base = self.rn_base[::direction_ind]
        #rn_base = rn_base.lstrip("0")

    def index(self, xi, yi, zi):
    
        # fill to 64
    
    
        # using only x,y (for 2D masks) find appropriate index and its value.
        mat_num = self.rn_base[((xi+4)*8)+(yi+4)] ## indexing of 8/8/8 cube.. [-3,4] 
        #x = x[(xi+4)+(yi+4)*8] ## indexing of 8/8/8 cube.. [-3,4] 
        # check indexing...
        mat_num = int(mat_num)
        return mat_num
