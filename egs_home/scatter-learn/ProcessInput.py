import numpy as np
class VoxScoper():
    def __init__(self, npad):
        self.npad = npad
        self.f = npad+1
        self.frame_half_width_right = self.f//2
        self.frame_half_width_left = self.f - self.frame_half_width_right

        from functools import partial
        self.partial=partial

    def _ScopeVox(self, s, inds):
        xind = inds[0]
        yind = inds[1]

        fhr = self.frame_half_width_right
        fhl = self.frame_half_width_left
        return s[xind-fhr:xind+fhl, yind-fhr:yind+fhl]
       
    def ScopeVox(self, s, xind="", yind="", num=""):

        try:
            return self._ScopeVox(s, (xind, yind))

        # xind and/or yind are non-ints
        except TypeError:
            try:
                xind = xindf(num)
                yind = yindf(num)
    
                return self._ScopeVox(s, (xind, yind))
    
            except TypeError:
                print("You must specify voxel to scope")
                raise TypeError


    def ScopeObj(self, s, sq):
        from itertools import product as itt_product
        ScopeVox_on_s = self.partial(self._ScopeVox, s)
        
        # Start at npad because that's where the object begins.
        x_range = range(self.npad, sq + self.npad)
        y_range = range(self.npad, sq + self.npad)
        all_inds = itt_product(x_range, y_range)

        s_dilated = map(ScopeVox_on_s, all_inds) 
        s_dilated = list(s_dilated)

        #s_dilated = []
        #for xind in range(self.npad, sq + self.npad):
        #    for yind in range(self.npad, sq + self.npad):
        #        s_dilated.append(self.ScopeVox(s, xind=xind, yind=yind))
    
        return s_dilated

## Making new scoped data ##
def _ProcessInput(sid, npad):
    # Take first slab for scattering object.
    #  - (K,512,4) = (K,8*8*8,f*f)?
    #  - ,:64 : only the first slab, for each sample.
    #  - ,-1 : only the material number
    N = sid.shape[0]
    #print(f"N: sid.shape[0] = {N}")
    n_blocks_in_front = round(sid.shape[1]**(2/3))
    sid = sid[:,:n_blocks_in_front,-1] 

    # putting each object into a square.
    sq = int(np.sqrt(sid[0].shape[-1]))
    sid = sid.reshape((N, sq, sq))
    #sid = np.array([0,-1,1])[np.int8(sid)]

    #sid = (np.pad(sidi,npad,constant_values=(-0.000001)) for sidi in sid)
    sid = [np.pad(sid_i,npad,constant_values=(0)) for sid_i in sid]
    #sid = np.array(sid)
    return sid, sq

def ProcessInput(sid, npad):
    sid, sq = _ProcessInput(sid, npad)
    # scope data
    voxscoper = VoxScoper(npad)
    ssc = [voxscoper.ScopeObj(s, sq) for s in sid]
    ssc = np.array(ssc) # scoped data 
    sid = np.array(sid)

    # ssX are shaped like (n_examples, n_pixels*scope_fame)
    #   > n_pixels = f*f
    #   > scope_frame = (npad+1)**2
    #ssc = ssc.reshape((len(lv),-1))
    return ssc, sid

def ProcessWindow(sid, npad):
    ssc, sid = ProcessInput(sid, npad)
    frame = sid.shape[-2]
    sid = sid.reshape((-1, 1, frame, frame))
    sid = NumpyToTorch(sid)
    return sid
    


def NumpyToTorch(x):
    import torch
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    x = torch.FloatTensor(x)
    return x

def RemoveCentralPixel(data):
    # (N_scoped_items, f*f pixels)
    middle_index = data.shape[-1]//2

    data_left = data[:, :middle_index]
    data_right = data[:, middle_index+1:]
    data = np.hstack((data_left, data_right))
    return data
