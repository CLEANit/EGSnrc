#from sklearnex import patch_sklearn
#patch_sklearn()
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import sklearn as sk
import sys
import os
import time
sys.path.append(os.getcwd())

#from scope_pattern import unscope_single
from LoadRecentData import LoadRecentData

random_seed = 12344
np.random.seed(random_seed)
random.seed(random_seed)

# Padding zeros for scope context
#try:
#    npad = sys.argv[2]
#    npad = int(npad)
#except:
#    npad = 2
    ## npad of 2 gives a input (scope) of 3x3.
    #
    #   - improvement in performance
    #     shows that deviation depends primarily on
    #     nearest neighbours. (6 nn)
    #
    #       > why would MORE infomation 
    #         reduce accuracy???
npad = 2
try:
    run_directory = sys.argv[1]
    ncase= sys.argv[2]
    ncase= int(ncase)
    sigma=sys.argv[3]
    sigma = float(sigma)
    
    N_sample=sys.argv[4]
    N_sample = int(N_sample)
    print(f"ncase:{ncase}",
          f"sigma:{sigma}",
          f"N_sample:{N_sample}")

    dc, sid, mats, \
    ncase_bk, sigma_bk = LoadRecentData(
                            data_path=run_directory,
                            history_index=0
    )
    if ncase != ncase_bk:
        raise ValueError

except IndexError:
    run_directory=None
    run_directory="_-_".join((
                    "run",
                    "ncase_10000",
                    "sigma_0.1",
                    "Nsample_1000"))
    dc, sid, mats, \
    ncase, sigma = LoadRecentData(data_path=run_directory,
                                  history_index=0)
    N_sample = dc.shape[0]


N = dc.shape[0]
xdim = round(sid.shape[1]**(1/3))

# Frame, so that the no scope doesn't contain at least 
# one pixel from ROI
f = npad+1

from ProcessInput import ProcessInput
ssc, sid = ProcessInput(sid, npad)

from TrainMod import SplitTestTrain, TrainModel

# For scoped models
input_data_train, \
input_data_test, \
tri, tei = SplitTestTrain(ssc, N)

# For un-scoped net.
#input_data_train, \
#input_data_test, \
#tri, tei = SplitTestTrain(sid,N)

# Choose the amount of data to use.
#each_nth = min((20, tri.__len__(), tei.__len__()))//2
#tri = tri[::each_nth]
#tei = tei[::each_nth]


#Reshape scope objects to N*m*m*f*f. 
# Easier to select relevant data only..
from ProcessLabelData import ProcessLabelData
dr = dc[tri]
de = dc[tei]

# Test/train split is done at the per-mask level. 
#   > Overlap between scoped views possible
#lv, lv_min, lv_minmax = ProcessLabelData(dc)
#labr = lv[tri].flatten()
#le = lv[tei].flatten()

#labr_min = 0
#labr_minmax = 1

# Normalize training data.
labr, labr_min, labr_minmax = ProcessLabelData(dr)

# Normalize test data using the same values
labe = ProcessLabelData(de,
                        lv_min=labr_min,
                        lv_minmax=labr_minmax)

labr = labr.flatten()
labe = labe.flatten()

## HYPER-PARAMETER OPTIMIZATION

# Cost-complexity was optimized for.
# Stacked Bootrap
#ccp_alpha = 0.022543969013557597

# Random Forest Regressor w/ [-1,1] normalization w/ mean.
#ccp_alpha = 0.00021876

# RFR w/ 0-1 normalization using -min
ccp_alpha = 0.0001329 

#from TrainMod import HyperParameterOptimizer
#Optimizer = HyperParameterOptimizer() 
#ccp_optif = lambda x: 1 - Optimizer.TestVal(input_data_train,
#                                            labr,
#                                            input_data_test,
#                                            le,
#                                            ccp_alpha=x)[1]
#from scipy import optimize as optim
#bnds = [[0,1]]
#ccp_opt = optim.minimize(ccp_optif,
#                         ccp_alpha,
#                         bounds=bnds)
#ccp_alpha = Optimizer.OptimalValue()[0]



## FOR SCOPED RFR SURROGATE
#mod = TrainModel(input_data_train[::10], labr[::10],
mod = TrainModel(input_data_train, labr,
                     ccp_alpha=ccp_alpha)

## FOR NON-SCOPED CNN SURROGATE
#from TrainNet import TrainNet
#mod = TrainNet(input_data_train, labr)



r2 = mod.score(input_data_test, labe)
oe = mod.predict(input_data_test)

#from ResidualModel import ResidualModel
#base_rfr = sk.ensemble.RandomForestRegressor(
#                               ccp_alpha=ccp_alpha
#)
#res_rfr = sk.ensemble.RandomForestRegressor()
#res_mod = ResidualModel(base_mod=base_rfr,
#                        residual_mod = res_rfr)
#res_mod.fit(input_data_train, labr)

print("DONE TRAINING")
print("r2: ",r2)

test_label_image = labe.reshape((-1, xdim, xdim))
test_output_image = oe.reshape((-1, xdim, xdim))

img_ind = np.random.randint(0,len(sid[tei]))

try:
#if True:
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4,figsize=(20,5))
    #fig, (ax3,ax4) = plt.subplots(1, 2,figsize=(20,5))
    fig.suptitle("".join((
                      "Voxelized Model Stages Testing: ",
                      f"{round(r2,3)}"))
    )

    from ColorbarFuncs import MakeMaterialsColorbar
    mats_norm, mats_cmap, \
    mats_mappable, mats_ticks = MakeMaterialsColorbar(
                                        input_data_train,
                                        input_data_test,
                                        fig)
                                        #cbar1_ax)
    #cb1.set_ticklabels(mats)

    
    # Input image
    input_data_image_width = input_data_test.shape[-1]
    input_data_image_width = np.sqrt(input_data_image_width)
    input_data_image_width = int(input_data_image_width)
    inp_date_plot = input_data_test.reshape(
                                    (-1,
                                     input_data_image_width,
                                     input_data_image_width)
    )
    #inp_date_plot = input_data_test.squeeze()
    assert inp_date_plot.shape[-1] == (npad + 1)
    inp_image = np.vstack(
                  [np.hstack(
                    [inp_date_plot[j*xdim + i] \
                               for i in range(xdim)])
                                     for j in range(xdim)])
                    #[inp_date_plot[img_ind][j][i] \
                    #           for i in range(npad+1)]) \
                    #                 for j in range(npad+1)])
    ix = f//2 + 1
    insh = inp_image.shape[0]
    lines = np.dstack((
                np.vstack((np.arange(0,insh,f)+0.5,
                           np.zeros(insh//f))).T-0.5,
                np.vstack((np.arange(0,insh,f)+0.5,
                           np.ones(insh//f)*insh)).T-0.5
    ))
    
    diff = (test_label_image - test_output_image)
    
    
    for line_i in lines:
        ax2.plot(line_i[0,:]-0.5, #in y
                 line_i[1,:],
                 color='w') 
        ax2.plot([-0.5,insh], # in x
                  np.array([1,1])*(ix)+0.5,
                  color='w') 
        ix += f
    #
    #inp_image = np.hstack(
    #               [np.vstack(xs_i) for xs_i \
    #                       in inp_date_plot[0]]
    #)
    
    ax2.imshow(inp_image,
               norm=mats_norm,
               cmap=mats_cmap)
    
    ax2.set_title("input")

    mats_images = (ax1, ax2)
#else:
except Exception as e:
    print(f"Exception: {e}")
    # Close failed plot attempt
    plt.close()


    fig, (ax1,ax3,ax4) = plt.subplots(1,
                                      3,
                                      figsize=(20,5))

    fig.suptitle("".join(("CNN Testing: ",
                          str(round(r2,3))
                    ))
    )

    #cbar1_ax = fig.add_axes([0.15,0.045,0.25,0.05])
    
    from ColorbarFuncs import MakeMaterialsColorbar
    mats_norm, mats_cmap, \
    mats_mappable, mats_ticks = MakeMaterialsColorbar(
                                            input_data_train,
                                            input_data_test,
                                            fig)
    #cb1.set_ticklabels(mats)

    mats_images = [ax1]


# Object image
ax1.set_title("object")
im1 = ax1.imshow(sid[tei][img_ind],
                 norm=mats_norm,
                 cmap=mats_cmap)

# Untested
from ProcessCSV import GetSharedCBPos
cbar1_ax = GetSharedCBPos(mats_images)
cb1 = fig.colorbar(mats_mappable,
                   cax=cbar1_ax,
                   orientation="horizontal")
cb1.set_ticks(mats_ticks)
cb1.set_ticklabels(mats)
cb1.set_label("Material")

# Plot Output images.
from ProcessCSV import MakeCommonColorbar 
target_norm, target_cmap, \
target_mappable = MakeCommonColorbar(
                        test_label_image[img_ind],
                        test_output_image[img_ind])

ax3.set_title("Prediction")
im3 = ax3.imshow(test_output_image[img_ind],
                 norm=target_norm,
                 cmap=target_cmap)

ax4.set_title("Label")
im4 = ax4.imshow(test_label_image[img_ind],
                 norm=target_norm,
                 cmap=target_cmap)

# Set position of color bar
cbar2_ax = GetSharedCBPos((ax3, ax4))
#cbar2_ax = fig.add_axes([0.475,0.045,0.38,0.05]) 

# Write Colorbar
cb2 = fig.colorbar(target_mappable,
                   cax=cbar2_ax,
                   orientation="horizontal")
cb2.set_label(r"Normalized Absorbed Dose [$Gy\cdot{cm}^2$]",
              loc='center')

# Plot difference plot
#ax5.set_title("difference")
#im5 = ax5.imshow(diff[img_ind],
#                 norm=target_norm,
#                 cmap=target_cmap)
##cbar3_ax = fig.add_axes([0.774,0.045,0.12,0.05])
##cb3 = fig.colorbar(im5,
#                    cax=cbar3_ax,
#                    orientation="horizontal")
##fig.savefig("rfr_modes_-_diff.png")


image_name= "_-_".join((f"work_flow",
                        f"ncase_{ncase}",
                        f"sigma_{sigma}",
                        f"Nsample_{N_sample}.png"))

fig_size = fig.get_size_inches()
fig.set_size_inches(fig_size[0],
                    h=fig_size[1]*1.25,
                    forward=True)
#fig.savefig(image_name)
plt.show()
plt.close()


## PREDICTED VS. EXPECTED ##
plt.figure()
plt.scatter(labe, oe)
plt.title(
    f"Predicted vs. Labelled Normalized Dose: {round(r2, 3)}"
)
plt.xlabel("Label")
plt.ylabel("Prediction")

range_min = min(np.sign(np.min(labr)),0)
range_max = 1

plt.xlim([range_min, range_max])
plt.ylim([range_min, range_max])

line = np.linspace(range_min, range_max, 100)
plt.plot(line, line, c='k')

evp_name= "_-".join((
        "evp",
        f"ncase_{ncase}",
        f"sigma_{sigma}",
        f"Nsample_{N_sample}.png"
))
#plt.savefig(evp_name)
plt.show()
#plt.close()

## Save Mod
from pickle import dump as pickle_dump

mod.labr_min = labr_min
mod.labr_minmax = labr_minmax
#mod_name = "_-_".join((
#                    "mod",
#                    f"ncase_{ncase}",
#                    f"sigma_{sigma}",
#                    f"Nsample_{N_sample}.pkl"
#))
mod_name = "trained_model.pkl"
with open(mod_name,'wb') as fil:
    pickle_dump(mod, fil)

## Worth noting: 
#  maximum values originating from where central voxel is 0.
#   > The max of this is undoubtably 1
#   > The mean is very close to the np.max(oe)
#       * since predictions end up being based on /average/
#         values under that type of pixel.
#zero_centrals= labr[np.dstack(
#                       np.where(input_data_train[:,4] == 0)
#)]



#with open("variety_records.txt",'a') as fil:
def GetModelType(mod):
    model_type = mod.__class__
    model_type = str(model_type)
    model_type = model_type.split("\'")
    model_type = model_type[1]

    model_type = model_type.split(".")
    model_type = model_type[-1]
    #model_type = mod.__class__()
    #model_type = str(model_type)
    #model_type = model_type.replace("()",'')
    return model_type

model_type = GetModelType(mod)
#with open(f"{model_type}_r2.csv",'a') as fil:
#    fil.write(f"{ncase},{sigma},{N_sample},{r2}")
#    fil.write("\n")
#
def PlotAllObjs():
    for i in range(sid.shape[0]):
        fig, axes = plt.subplot_mosaic("AB")
        axes["A"].imshow(sid[i])
        axes["B"].imshow(lv[i])
        fig.show()
#si, single_output_image = FullProcessSingle(csv_path,
#                                            mod=mod)
#PlotSinglePair(csv_path, mod=mod)
