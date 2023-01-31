import numpy as np
import pandas as pd
import sklearn as sk
import pickle
import sys
import copy
import random
import itertools as itt
import functools as ft

#from ProcessInput import ProcessInput
from ProcessCSV import FullProcessSingle, CompareImages
from ProcessLabelData import ProcessDistData
import matplotlib.pyplot as plt
from copy import deepcopy


try:
    rn = sys.argv[1]

except IndexError:
    rn = "62783361878388904048055595911667886675"
    #rn = "252785838956249757503353786711590552658"
#    rn = "334800398373121594589956989"

input_object_path = "".join((f"./{rn}_-_detector_results/",
                             f"scatter-learn_-_num_0-{rn}_-_scatter_input.csv"))

from LoadModel import LoadModel, InferNPad, GetExpandedShape
mod = LoadModel(mod_path="trained_model_new.pkl")
lr_min = mod.labr_min
lr_minmax = mod.labr_minmax

# Calculate npad from model
npad = InferNPad(mod)


input_dist_data = FullProcessSingle(input_object_path, npad)
idd_og = copy.deepcopy(input_dist_data)
dist_pred = mod.predict(input_dist_data)

distribution_path = "".join((
                    f"./{rn}_-_detector_results/",
                    "column_3.csv"))

# (f*f pixels, N experiments)
dist_lab = pd.read_csv(distribution_path, header=None)
dist_lab = dist_lab.to_numpy()
dist_lab_og = dist_lab.copy()
#
dist_lab = ProcessDistData(dist_lab, lv_min=lr_min, lv_minmax=lr_minmax)
dist_lab = dist_lab.squeeze()
central_vox=4
dist_input = np.array([input_dist_data[:,central_vox] for x in range(len(dist_lab))])







from LoadRecentData import LoadRecentData
#run_directory=None
run_directory="run_-_ncase_10000_-_sigma_0.1_-_Nsample_1000"
dc, sid, mats, ncase, sigma = LoadRecentData(data_path=run_directory,
                                             history_index=0)
N_sample = dc.shape[0]

#
#
#print("NPAD: ",npad)

sid_og = sid.copy()

N = dc.shape[0]
xdim = round(sid_og.shape[1]**(1/3))
# Frame, so that the no scope doesn't contain at least one pixel from ROI
f = npad+1

from ProcessInput import ProcessInput
#sid /= np.max(sid)
ssc, sid = ProcessInput(sid, npad)

from TrainMod import SplitTestTrain, TrainModel

# For scoped models
input_data_train, input_data_test, tri, tei = SplitTestTrain(ssc, N)

# For un-scoped net.
#input_data_train, input_data_test, tri, tei = SplitTestTrain(sid, N)

# reshape scope objects to N*m*m*f*f. Easier to select relevant data only..
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

labr, labr_min, labr_minmax = ProcessLabelData(dr)
labe = ProcessLabelData(de, lv_min=labr_min, lv_minmax=labr_minmax)

labr = labr.flatten()
labe = labe.flatten()

#print("Begin Training")
#mod.fit(input_data_train, labr)

predr = mod.predict(input_data_train)
prede = mod.predict(input_data_test)
pred_dist = mod.predict(input_dist_data)

material_numbers = np.unique(input_data_test)

def R2_by_mat(mat_number, in_dat_e, prede, labe):
    central_points = in_dat_e[:, in_dat_e.shape[-1]//2]
    mat_cp = np.where(central_points == mat_number)[0]

    mat_pred = prede[mat_cp]
    try:
        mat_lab = labe[:,mat_cp]
        #mat_lab = mat_lab.flatten()
    except IndexError:
        mat_lab = labe[mat_cp]

    return central_points, mat_cp, mat_pred, mat_lab

def MakeBoxPlot(material_numbers, in_dat_e, prede, labe):
    #cp, mat_cp, mat_pred, mat_lab = R2_by_mat
    fig, ax = plt.subplots(1)
    mp_l = []
    ml_l = []
    for mat_number_i in material_numbers:
        cp0, mat_cp0, mp0, ml0 = R2_by_mat(mat_number_i, in_dat_e, prede, labe)

        fig_suptitle = "Deviation Between Prediction and Simulation by Material"
        if len(ml0.shape) > 1:
            print(f"ml0 shape:{ml0.shape}")
            ml0 = ml0[0]
            fig_suptitle = "Deviation Between Prediction and a Single Voxel From Repeated Simulation"

        ml_l.append(ml0)
        mp_l.append(mp0[0])

    bp = ax.boxplot(ml_l)#, notch=mp_l)
    for bp_ind, med in enumerate(bp['medians']):
        x,y = med.get_data()
        y = [mp_l[bp_ind],mp_l[bp_ind]]
        plt.plot(x,y,'r')
    ax.set_xticklabels(mats)
    ax.set_xlabel("Material")
    ax.set_ylabel("Normalized Absorbed Dose")
    fig.suptitle(fig_suptitle)
    fig.savefig("material_boxplot.png")
    #fig.savefig("material_boxplot_single_voxel.png")
    
    #fig.legend(["a","b",'c','d','e','f'])
    #fig.legend(["","",'','','','f'])
    fig.show()

#plt.show()

#r2_mat0 = sk.metrics.r2_score(mp, ml)
cpd, mtcpd, mpd, mld = R2_by_mat(1, input_dist_data, pred_dist, dist_lab)
cpe, mtcpe, mpe, mle = R2_by_mat(1, input_data_test, prede, labe)

#MakeBoxPlot(material_numbers, input_dist_data, pred_dist, dist_lab)
MakeBoxPlot(material_numbers, input_data_test, prede, labe)


# Compare histograms from each set
rep_vox_ind = 27 
rep_vox = dist_lab[:, rep_vox_ind]
rep_mat = input_dist_data[rep_vox_ind, central_vox]

repeated_mat = list(set(dist_input[:,rep_vox_ind]))

# Should only be one material in the superset
assert len(repeated_mat) == 1

# Material in the super set should be the same as that of the superset...
assert rep_mat == repeated_mat[0]

# Get repeated indicies and values for material type
cpr, mtcpr, mpr, mlr = R2_by_mat(rep_mat, input_data_train, predr, labr)
cpe, mtcpe, mpe, mle = R2_by_mat(rep_mat, input_data_test, prede, labe)
superset_mats = np.unique(input_data_test[mtcpe, central_vox])

# Verify same materials
assert rep_mat == superset_mats[0]


plt.close()

plt.figure()
plt.title(f"Repeatedly Simulated Voxel {rep_vox_ind} and {mats[rep_mat].capitalize()} Superset Distributions")
hd, hb = np.histogram(mle, density=True)#, color="orange", alpha=0.8, ec="orange")
plt.close()
#
plt.figure()
plt.title(f"Repeatedly Simulated Voxel {rep_vox_ind} and {mats[rep_mat].capitalize()} Superset Distributions")
plt.hist(mlr, bins=hb, density=True, color="red", alpha=0.8, ec="red")
plt.hist(mle, bins=hb, density=True, color="orange", alpha=0.8, ec="orange")
plt.hist(rep_vox, bins=hb, density=True, color="blue", alpha=0.8, ec="blue")
#plt.hist(rep_vox, density=True, color="blue", alpha=0.20, ec="blue")
plt.legend(["Repeated Voxel",'Material Test', 'Material Train'])
plt.xlabel("Normalized Absorbed Dose")
plt.ylabel("Probability Density")
#plt.savefig(f"rep_vox_{rep_vox_ind}_-_vs_-_mat.png")
plt.show()

