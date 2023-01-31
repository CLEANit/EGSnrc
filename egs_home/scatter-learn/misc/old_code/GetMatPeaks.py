

def GetMatPeaks(sid_og, lv, tri):
    '''Get average value for pixels "below" a \
        particular material type'''

    from ProcessInput import _ProcessInput
    import numpy as np


    sid_narrow, sq_narrow = _ProcessInput(sid_og, 0)
    sid_narrow = np.array(sid_narrow)

    sid_narrow = sid_narrow[tri]
    lr = lv[tri]

    peaks = np.unique(sid_narrow)

    calculate_peak = lambda mat_i: np.mean(lr[np.where(sid_narrow == mat_i)])

    peaks = map(calculate_peak, peaks)
    peaks = list(peaks)
    return np.array(peaks)




#from ProcessLabelData import ProcessLabelData
#lv = ProcessLabelData(dc)
#N = lv.shape[0]
#xdim = round(sid_og.shape[1]**(1/3))
##

# ncase | xdim | ydim | zdim | xpos | ypos

## Find the two maximums
#l = lv.copy()
#l = np.sort(l).flatten()
#lhd,lhb = np.histogram(l,bins=100,density=True)
### Split point
#li = (np.where(np.diff(lhd==0)==True)[0]+1).tolist()
#
### Weighted averages of each half.. 
##   - Should this done for each image?
##       > Wouldn't this lose ability to 
##         discriminate magnitude between 
##         images?
##
#w1 = lhd[:li[0]]/np.sum(lhd[:li[0]])
#w2 = lhd[li[1]:]/np.sum(lhd[li[1]:])
#m1 = np.dot(w1,lhb[:li[0]]) #lower max
#m2 = np.dot(w2,lhb[:-1][li[1]:]) #upper max


## sid2 is object,
#   equivalent to sid3 = np.array([m1,m2])[np.int8(sid2)]
#  'normalized' between max(lable),min(lable)
#sid2 = np.array([m2,m1])[np.int8(sid2)]

#sid2 *= (m1-m2)
#sid2 += m2

#sid2 *= (m2-m1)
#sid2 += m1

#sid2 = np.abs(1-sid2) #flip for shadow

#lv -= sid2
#lv /= sid2 # relative difference ?
#lv = lv - sid2

#max_side = np.sign(np.max(lv) - np.min(lv))
#lv -= max_side

