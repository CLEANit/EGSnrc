import numpy as np
import h5py as h5

def parse(fli):
    so = np.array([list(map(int,x.split(",")[0:3])) for x in fli.split("\n")[:-1]])
    ##^ scatter object, raw. (n**3, 3)
    v = np.array([list(map(int,x.split(",")[3]))[0] for x in fli.split("\n")[:-1]])#.reshape((-1,4))
    ##^ materials vec (n,)
    xs = np.array([[sum(v[np.intersect1d(np.where(so[:,0]==x)[0],np.where(so[:,1]==y)[0])]) for x in np.unique(so[:,0])] for y in np.unique(so[:,1])],dtype=bool)
    xs = np.int8(xs)
    v = v.reshape((-1,4))
    xs += np.ones_like(xs)
    ## xs - cross-section.
    return so, v, xs


def scope_single(x,y,t,dx,dy): ## Not used now.
    #ts = t.copy()
    #ts[0,0]=4
    #ts[0,-1]=4
    #ts[-1,0]=4
    #ts[-1,-1]=4
    #ts[y,x]=3
    return t[y-dy//2:+y+dy//2+1, x-dy//2:x+dx//2+1]

#with h5.File("croped_scoped_data_binary.hdf5",'r') as hf:
    # ['detector_results', 'detector_results_data_table', 'scatter_input', 'scatter_input_data_table', 'scope_data', 'scope_data_coords', 'scope_data_labels']
#    dt = hf['detector_results_data_table'][:].copy()
#    d = hf['detector_results'][:].copy()
    #sdc = hf['scope_data_coords'][:].copy()

def scope(si,t,fx,fy,fwx,fwy):
    with open(si,'r') as f:
        fl = f.read()
    so, v, xs = parse(fl)

    scope_data = np.array([[t[y-fy//2:+y+fy//2+1, x-fx//2:x+fx//2+1] for x in range(fx//2,fx//2+fwx)] for y in range(fy//2,fy//2+fwy)])   #scoped data. (10,10)
    scope_data_coords = np.array([[[x,y] for x in range(fx//2,fx//2+fwx)] for y in range(fy//2,fy//2+fwy)])   #scoped data. (10,10)
    return scope_data, scope_data_coords

def get_scope_data():
#if True:
    with open("scatter_input_list",'r') as f:
        fl = f.read()
    fl = fl.split("\n")#.remove('')#[1:-1]
    fl.remove('')


    fi = fl[0]
    with open(fi,'r') as f:
        fli = f.read()
    so, v, xs = parse(fli)
    #xs += 1
    
    pat_xd = xs.shape[0]
    pat_yd = xs.shape[1]
    nx= 16## Arbitrary.
    ny=nx## Arbitrary.
    fx = (nx-pat_xd)//2+1
    fy = (ny-pat_yd)//2+1

    fwx = (nx - fx) + 1
    fwy = (ny - fy) + 1


    zero_ind = nx//2
    t = np.zeros((nx,ny))
    #t[0,0]=-1
    #t[0,-1]=-1
    #t[-1,0]=-1
    #t[-1,-1]=-1
    t[zero_ind-pat_xd//2:zero_ind+pat_xd//2,zero_ind-pat_yd//2:zero_ind+pat_yd//2] = xs ## central according to 16.
    x = np.array([scope(flj,t,fx,fy,fwx,fwy)[0] for flj in fl])
    xd = np.array([scope(flj,t,fx,fy,fwx,fwy)[1] for flj in fl])
    return x,xd
#x,c = get_scope_data()

def unscope_single(sd):
    n=sd.shape[-1]
    N = sorted(set(sd.shape))[1] #Assuming n_trial > n_scope > scope_dim (last cond not optional)
    #print(sorted(set(sd.shape))) #Assuming n_trial > n_scope > scope_dim (last cond not optional)
    #print(N)
    #N = sd.shape[-2]
    y = np.empty((N+n-1,N+n-1))
    for i in range(N):
        for j in range(N):
            y[j:j+n,i:i+n] = sd[j][i]#*N+j]
    return y
#y = unscope_single(x[0])
#for l in y:
#    print(l)


    #> Should pat_xd and pat_yd be switched?
    #ts = t.copy()
    #y = fy //2
    #x = fx //2
    #dy = fy
    #dx = fy
    #ts[y,x]=3
    #a = ts[y-dy//2:+y+dy//2+1, x-dy//2:x+dx//2+1]
