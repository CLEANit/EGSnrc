import numpy as np

def ProcessLabelData(lv, lv_min=None, lv_minmax=None):
    #lv = dc.copy()
    # Predicting energy deposited
    lv = lv[:,:,2]
    N = lv.shape[0]

    detect_f = int(np.sqrt(lv.shape[-1]))

    # Crop detector data to relevant section.
    lv = lv.reshape((N, detect_f, detect_f)) #shape: (N,f*f) -> (N,f,f)

    # cropped
    mid = lv.shape[-2]//2
    xdim = 8 #lv.shape[-2]
    #xdim = round(sid.shape[1]**(1/3)) #shape taken from dv: only considering directly "below" obj

    # Crop in y/x
    lv = lv[:,mid-xdim//2:mid+xdim//2,:] 

    # Crop in x/y
    lv = lv[:,:,mid-xdim//2:mid+xdim//2] 
    
    if lv_min is None:
        lv, lv_min, lv_minmax = NormalizeLabel(lv)

        return lv, lv_min, lv_minmax

    else:
        lv = NormalizeLabel(lv, lv_min=lv_min, lv_minmax=lv_minmax)

        return lv

def NormalizeLabel(lv_og, lv_min=None, lv_minmax=None):
    lv = lv_og.copy()
    if lv_min is None:
        lv_min = np.mean(lv)
        #lv_min = np.min(lv)

        lv -= lv_min


        #lv_minmax = lv.flatten()[np.argmax(np.abs(lv))]
        #lv /= np.abs(lv_minmax)

        lv_minmax = np.max(np.abs(lv))
        #lv_minmax = np.max(lv)
        lv /= lv_minmax

        #lv *= 2
        #lv -= 1
        #lv -= np.sign(lv_minmax)
        return lv, lv_min, lv_minmax

    else:
        lv -= lv_min
        lv /= lv_minmax

        #lv *= 2
        #lv -= np.sign(lv_minmax)
        return lv

    
    # Not helpful for -mean(lv)
    #lv *= 1.1 ## Helps ?? why?
    # *1.5 - *2 : 0.916
    # *1.1 : 0.915
    # *1.0 : 0.914

    #lv -= 1
    #return lv, lv_min, lv_minmax

def ProcessDistData(lv, lv_min=None, lv_minmax=None):
    #lv = dc.copy()
    # Predicting energy deposited
    lv = lv[:,3:]
    lv = lv.T
    N = lv.shape[0]

    detect_f = int(np.sqrt(lv.shape[-1]))

    # Crop detector data to relevant section.
    #lv = lv.reshape((N, detect_f, detect_f, -1)) #shape: (N,f*f) -> (N,f,f)
    lv = lv.reshape((N, detect_f, detect_f)) #shape: (N,f*f) -> (N,f,f)

    # cropped
    mid = detect_f//2
    xdim = 8 #lv.shape[-2]
    #xdim = round(sid.shape[1]**(1/3)) #shape taken from dv: only considering directly "below" obj

    # Crop in y/x
    lv = lv[:, mid-xdim//2:mid+xdim//2, :] 

    # Crop in x/y
    lv = lv[:, :, mid-xdim//2:mid+xdim//2] 

    # flatten
    lv = lv.reshape((N, xdim*xdim))
    if lv_min is None:
        pass
    else:
        lv = NormalizeLabel(lv, lv_min=lv_min, lv_minmax=lv_minmax)

    return lv
    

