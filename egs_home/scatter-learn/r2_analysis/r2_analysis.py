import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn import metrics
sys.path.append("/home/user/school/res/EGSnrc/egs_home/scatter-learn/")



#csv_file_path = "combine_r2_measures/r2_combine.csv"
csv_file_path = "rfr_r2_combo.csv"
#csv_file_path = "RandomForestRegressor_r2.csv"
#csv_file_path = "sktorch_r2.csv"#"model_r2.csv"
data_df = pd.read_csv(csv_file_path, header=None)
data = data_df.copy()
data = data.to_numpy()

N_data_set = np.unique(data[:,2])

#def linfit(ZZ,YY, j):
#    z = ZZ[:,j]
#    y = YY[:,j]
#    nzi = z.nonzero()
#    z = z[nzi]
#    y = y[nzi]

def linfit(y,z):
    p = np.polyfit(y,z,1)
    fit = lambda xi: xi*p[0] + p[1]
    return fit
    

for N_data in N_data_set:
    print(f"N_data: {N_data}")
    data = data_df.copy()
    data = data.to_numpy()
    data = data[data[:,2] == N_data]
    #data = pd.read_csv(csv_file_path)
    
    model_type = csv_file_path.split("\/")[-1]
    model_type = model_type.replace("_r2",'')
    model_type = model_type.replace("r2_",'')
    model_type = model_type.replace(".csv",'')
    
    #XX = data[:,:2]
    XX = data[:,0]
    YY = data[:,1]
    #XX = XX.reshape((1,-1))
    YY = YY.reshape((1,-1))
    
    Xu = np.unique(data[:,0])
    Yu = np.unique(data[:,1])
    XX, YY = np.meshgrid(Xu, Yu)
    
    ZZ = np.empty_like(XX)
    
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            dx = data[data[:,0] == XX[i,j]][:,-1]
            dy = data[data[:,1] == YY[i,j]][:,-1]
            dxy = np.intersect1d(dx,dy)
            #dxy = dxy[-1]
            try:
                ZZ[i,j] = dxy
            except ValueError:
                ZZ[i,j] = 0

    # incidents
    zi = np.where(ZZ == 0)
    zi = np.dstack(zi)
    zi = zi[0]

    #for i,j in zi:

    #    x = XX[i]
    #    zx = ZZ[i]
    #    nzix = zx.nonzero()
    #    x = x[nzix]
    #    zx = zx[nzix]

    #    fx = linfit(x,zx)
    #    zfx = fx(XX[i,j])

    #    y = YY[:,j]
    #    zy = ZZ[:,j]

    #    nziy = zy.nonzero()
    #    y = y[nziy]
    #    zy = zy[nziy]


    #    y0 = YY[i,j]
    #    fy = linfit(y,zy,y0)
    #    zfy = fy(y0)
    #    1/0
    ZZ_og = ZZ.copy()
    XX_og = XX.copy()
    YY_og = YY.copy()


    

    A = lambda v: np.matrix(np.dstack((np.ones_like(v[0]),*(vi**2 for vi in v)))[0])
    #A = lambda v: np.matrix(np.dstack((np.ones_like(v[0]),*v))[0])
    #A = lambda v: np.matrix(np.dstack(v)[0])

    #t = lambda v, zz: ((A(v).T*((np.cov(A(v)))**-1)*A(v))**-1)*A(v).T*(np.cov(A(v))**-1)*zz
    t = lambda v, zz, w: ((A(v).T*((w.T*np.cov(A(v))*w)**-1)*A(v))**-1)*(A(v).T)*((w.T*np.cov(A(v))*w)**-1)*zz
    #t = lambda v, zz: ((A(v).T*A(v))**-1)*A(v).T*zz

    min1 = lambda x: min(x,1)
    vmin1 = np.vectorize(min1)

    XX = np.log(XX)
    YY = np.log(YY)

    nzi_og = ZZ.nonzero()
    zi_og = np.where(ZZ==0)
    #w = 

    #ZZ_in = ZZ.copy()
    from copy import deepcopy
    def conv(ZZ_in, N=10):
    #if True:
        ZZ = deepcopy(ZZ_in)
        nzi = ZZ.nonzero()
        zz = ZZ[nzi]
        zz = np.matrix(zz).T
        xx = XX[nzi]
        yy = YY[nzi]

        w = np.ones_like(ZZ.flatten().nonzero()[0], dtype=float)
        w_og = ZZ_og.flatten().nonzero()
        try:
            w[w_og] = 1 + 1/(N+1)
        except IndexError:
            pass
        
        w /= np.sum(w)


        

        tfit = t((xx,yy),zz, w)
        
        xz = XX[zi_og]
        yz = YY[zi_og]

        z_pred = A((xz,yz))*tfit
        z_pred = np.array(z_pred.flatten())
        noise = np.random.uniform(low=-1, high=1,size=z_pred.shape)/(N+10)
        
        z_pred += noise
        ZZ[zi_og] = z_pred.flatten()
        ZZ = vmin1(ZZ)
        return ZZ
    #u=0.9

    #ZZ, zp = conv(ZZ)
    #ZZ_one = ZZ.copy()
    #ZZ_one = conv(ZZ_one)
    for dummy in range(1000):
        ZZ = conv(ZZ,N=dummy)
        #ZZ = (1-u)*ZZ + u*conv(ZZ,N=dummy)
    #1/0


    
    #XX = np.log10(XX)
    #YY = np.log10(YY)
    from ProcessCSV import MakeCommonColorbar
    max_cb_value = 1
    min_cb_value = 0.6
    
    max_like = np.ones_like(ZZ)
    max_like *= max_cb_value
    
    min_like = np.ones_like(ZZ)
    min_like *= min_cb_value
    
    range_of_values = np.dstack((min_like, max_like))
    norm, cmap, cb_mappable = MakeCommonColorbar(ZZ, range_of_values)
    
    N_data = int(N_data)
    fig = plt.figure()
    plt.title(rf"Model trained on {N_data}: $r^2$ by $\log_{{10}}$ N Particles and $\log_{{10}}$ Beam Energy $\sigma$")
    plt.contourf(XX, YY, ZZ, norm=norm, cmap=cmap)
    plt.xlabel(r"Log Number of Particles")
    plt.ylabel(r"Log Beam Energy Profile $\sigma$")
    cb1 = plt.colorbar(cb_mappable, orientation="horizontal")
    cb1.set_label(r"$r^2$")

    point_coords = np.dstack((XX,YY))
    point_coords = point_coords.reshape((-1, 2))
    plt.scatter(point_coords[:,0],point_coords[:,1], c='c', marker='x')
    
    point_coords = np.dstack((XX[nzi_og],YY[nzi_og]))
    point_coords = point_coords.reshape((-1, 2))
    plt.scatter(point_coords[:,0],point_coords[:,1], c='g', marker='x')
    figure_name = f"{model_type}_-_Nsample_{N_data}_-_r2_by_Ncase_vs_sigma.png"
    #plt.savefig(figure_name)
    fig.show()
    
    
    
    
