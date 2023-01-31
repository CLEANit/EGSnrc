#!/home/user/miniconda3/envs/skmod/bin/python
import numpy as np
from numpy import histogram
import pickle
import pandas as pd
import sys
import random
import itertools as itt
import functools as ft

#from ProcessInput import ProcessInput
from ProcessCSV import FullProcessSingle, CompareImages
from ProcessLabelData import ProcessDistData

import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import sklearn as sk

from copy import deepcopy

try:
    rn = sys.argv[1]

except IndexError:
    rn = "22792750316568805750554156087953945578"
    #rn = "62783361878388904048055595911667886675"
    #rn = "252785838956249757503353786711590552658"
#    rn = "334800398373121594589956989"
input_object_path = "".join((
                    f"./{rn}_-_detector_results/",
                    f"scatter-learn_-_num_0-{rn}_-_scatter_input.csv"))

from LoadModel import LoadModel, InferNPad, GetExpandedShape
mod = LoadModel()
lr_min = mod.labr_min
lr_minmax = mod.labr_minmax

# Calculate npad from model
npad = InferNPad(mod)

# Get test object input.
input_data = FullProcessSingle(input_object_path, npad)
prediction = mod.predict(input_data)

distribution_path = "".join((
                    f"./{rn}_-_detector_results/",
                    "column_3.csv"))

# (f*f pixels, N experiments)
label = pd.read_csv(distribution_path, header=None)
label = label.to_numpy()
label_og = label.copy()

label = ProcessDistData(label, lv_min=lr_min, lv_minmax=lr_minmax)
label = label.squeeze()






# r2 values between simulations
label_inds = range(len(label))
label_inds_perms = itt.permutations(label_inds, 2)
label_take = ft.partial(label.take, axis=0)
label_permus = map(label_take, label_inds_perms)
r2_gen = itt.starmap(sk.metrics.r2_score, label_permus)
r2_mean = np.fromiter(r2_gen, dtype=float)
r2_mean_from_gen = np.mean(r2_mean)
print(f"r2 from permutation generator, {r2_mean_from_gen}")



# Select test value
def PullOut(label_og, test_ind=None):
    label = label_og.copy()
    if test_ind is None:
        test_ind = np.random.randint(0, len(label))

    test = label[test_ind]
    label = np.vstack((label[:test_ind], label[test_ind+1:]))
    #label = np.concatenate((label[:test_ind], label[test_ind+1:]))
    return test, label


def WeedBadValues(label, def_func):
    bad_inds = def_func(label, axis=0)
    bad_inds = np.unique(bad_inds)

    # Have to weed from right to left, to preserve order
    bad_inds.sort()
    bad_inds = bad_inds[::-1]

    bad_val_l = []
    for bad_ind_i in bad_inds:
        bad_val, label = PullOut(label, test_ind=bad_ind_i)
        bad_val_l.append(bad_val)

    return label, bad_val_l


def Bootstrap(label):
    # Bootstrap here?
    bootstrap_frac = 3
    bootstrap_inds = np.arange(len(label))

    random.shuffle(bootstrap_inds)
    cutoff_ind = (bootstrap_frac - 1) * len(label)//bootstrap_frac

    bootstrap_inds = bootstrap_inds[:cutoff_ind]
    label = label[bootstrap_inds]
    return label, bootstrap_inds


def FindBin(val, bins):
    # DEV this caused a bug when it was just >
    upper = bins >= val
    upper = upper.tolist()
    
    ## Make sure value is within range
    #if val > np.max(bins):
    #    return len(bins) - 2
    #    #return -1
    #elif val < np.min(bins):
    #    return 0

     
    #try:
        # First True is upper bin edge of the right bin
    ind = upper.index(True)
    ind -= 1

    #except ValueError:
        # value exceeds largest bin
    #    ind = len(bins) - 1

    #ind = np.digitize(val, bins, right=True)
    return ind

def Binomial(bin_prob, bin_counts, total_counts):
    ## SHOULDN'T BE USED W/O BOOTSTRAPPING.
    #   > BIN_PROB IS NOT ACCURATE, SO SKEWS CALC.
    #   > PROB FOR SINGLE BINoRa
    #print(f"total_counts: {total_counts}, {total_counts.dtype}")
    bin_prob = bin_counts/total_counts
    other_counts = total_counts - bin_counts

    prob = np.math.factorial(total_counts)
    prob /=  np.math.factorial(bin_counts)
    prob /=  np.math.factorial(other_counts)
    #
    prob *=  ((bin_prob)**(bin_counts)) 
    prob *=  ((1-bin_prob)**(other_counts))

    #print(f"bin_prob: {bin_prob}")
    #print(f"bin_counts: {bin_counts}")
    #print(f"total_counts: {total_counts}")
    try: 
        assert bin_prob < 1
        assert bin_prob > 0 
        assert bin_counts < total_counts
    except AssertionError:
        1/0
    return prob

def _CountProb(val, bins, counts):
    total_counts = np.sum(counts)

    if val > np.max(bins):
        print(f"MAX: {val} > {np.min(bins)}")
        #bin_width = val - np.max(bins)
        total_counts += 1
        bin_counts = 1
        bin_prob = 1 / total_counts
        #bin_prob = 1 / (np.sum(counts))

    elif val < np.min(bins):
        #print(f"MIN: {val} > {np.min(bins)}")
        #bin_width = np.min(bins) - val
        total_counts += 1
        bin_counts = 1
        bin_prob = 1 / total_counts
        #bin_prob = 1 / (np.sum(counts))

    else:
        # Overwrite probability
        val_bin_ind = FindBin(val, bins) 
        bin_counts = counts[val_bin_ind]
        #bin_counts += 1

        bin_prob = bin_counts / total_counts
        #bin_prob = (bin_counts)/ (np.sum(counts))
        #
        #prob = CountProb(val, bins, counts)
        #bin_width = bins[val_bin_ind+1] - bins[val_bin_ind]
    
    # bin_counts (+1?) from a binomial distribution 
    #
    return bin_prob, bin_counts, total_counts

def CountProb(val, bins, counts):
    bin_prob, *_ = _CountProb(val, bins, counts)
    return bin_prob

def CountProbDensity(val, bins, counts):
    ## DEV PROBLEM
    #  All of the following integrate to 1.0, when (integrated ?) 
    #  np.dot'd by np.diff(bins)
    #
    #       pd_sum = (counts / np.sum(counts)) / np.diff(bins)
    #       pd_int = counts / np.dot(c, np.diff(b))
    #       pd_hist = histogram(label[:, value_index], density=True, bins=bins)
    #
    #  But they are also all different.
   
    #if type(counts[0]) != 
    bin_prob, bin_counts, total_counts = _CountProb(val, bins, counts)
    #prob = Binomial(bin_prob, bin_counts, total_counts)
    prob = bin_prob 
    #
    prob_density = prob / np.sum(counts * np.diff(bins))
    #prob_density = prob / bin_width
    #return prob_density
    #
    #
    #log_lk_sum_term = bin_counts * np.log(prob * total_counts)
    #return log_lk_sum_term

    return prob_density

def cpd2(val, label, voxel_ind):
    d, b = histogram(label[:, voxel_ind], bins=master_bins[voxel_ind], density=True)
    bin_ind = FindBin(val, b)
    val_density = d[bin_ind]
    return val_density



def FullHist(label_og, voxel_ind, test_ind=None, test=None):
    label = label_og.copy()
    if test is None:
        test, label = PullOut(label, test_ind=test_ind)

    label, bootstrap_inds = Bootstrap(label)

    counts, bins = histogram(label[:, voxel_ind], bins=master_bins[voxel_ind])
    counts, bins = AggregateEmptyBins(counts, bins)

    bin_prob, bin_counts, total_counts = _CountProb(test[voxel_ind], bins, counts)
    prob_dens = CountProbDensity(test[voxel_ind], bins, counts)

    pd2 = cpd2(test[voxel_ind], label, voxel_ind)
    return prob_dens, bin_prob, pd2, bin_counts, total_counts, bins, counts 

def GetProbs(label_og, voxel_ind, test_ind=None, test=None):
    pd, bp, pd2, *_ = FullHist(label_og, voxel_ind, test_ind=test_ind, test=test)
    return pd, bp, pd2

#def AggregateEmptyBins(bins, counts):
def AggregateEmptyBins(counts, bins):
    #    return counts, bins
    #    1/0
    full_inds = counts.nonzero()
    
    # bins are left?-edge leading
    #full_inds = np.hstack((full_inds, full_inds+1))
    #full_inds = full_inds.unique()

    bins_full = bins[full_inds]
    #if counts[-1] == 0:
    bins_full = np.hstack((bins_full, bins[-1]))


    counts_full = counts[full_inds]
    #counts, bins = np.histogram(label, bins=bins_full)
    return counts_full, bins_full










N_voxel = label.shape[-1]
all_vals = np.vstack((label, prediction))
master_bins = [histogram(all_vals[:,i])[1] for i in range(N_voxel)]

def CalcML(label_og, test): #, label=None):
    label = label_og.copy()
    #if label is None:
    #    label = label_og.copy()
    #    # Remove ... suspect? values
    #    #label, bad_val_l1 = WeedBadValues(label, np.argmax)
    #    #label, bad_val_l2 = WeedBadValues(label, np.argmin)
    #    # Select item to test
    #    #test, label = PullOut(label, test_ind=None)
    #    
    #    # Bootstrap values for historgram (Necessary??)
    #    #   > Only uses a bootstrap if test isn't provided??
    #    #label, bootstrap_inds = Bootstrap(label)
    if test is None:
        test, label = PullOut(label, test_ind=None)
        
    label_bsp, bootstrap_inds = Bootstrap(label)
    
    # Voxel-wise bin data
    #pixelwise_hists = (histogram(label_bsp[:, voxel_ind], bins=master_bins[voxel_ind], density=True) for voxel_ind in range(N_voxel))
    pixelwise_hists = (histogram(label_bsp[:, voxel_ind], bins=master_bins[voxel_ind]) for voxel_ind in range(N_voxel))
    pixelwise_hists = itt.starmap(AggregateEmptyBins, pixelwise_hists)

    testpix_and_hists = ((test[voxel_ind], b, c) for voxel_ind, (c, b) in enumerate(pixelwise_hists))

    test_prob_density = itt.starmap(CountProbDensity, testpix_and_hists)

    #mxli = math.prod(test_prob_density)
    mxli = np.fromiter(test_prob_density, dtype=float)
    mxli = np.log(mxli)
    mxli = np.sum(mxli)
    return mxli #, label.copy()


#def MakeMLDist(label_og, test=None):
#    #label = label_og.copy()
#    
#    
#    # Calc Liklihood
#    #li = CalcML(label, test)
#
#    # Take Log Likelihood
#    return li #, test, lb

def SampleML(label):
    SampleML_part = lambda x: CalcML(label, test=None)
    N_iter = 10000
    ml_sample = map(SampleML_part, range(N_iter))
    ml_sample = tqdm(ml_sample, total=N_iter)

    #for itr in range(N_iter):
    #    #label2 = deepcopy(label.copy())
    #    test, label2 = PullOut(label2, test_ind=None)
    #    li, lb = CalcML(label2.copy(), test.copy())
    #    li = np.log(li)

    #    del label2, test, li, lb
    #for tup in ml_sample:
        #ml_l.append(mli)

    #ml_sample = ml_l
    ml_sample = np.fromiter(ml_sample, dtype=float) 
    
    # Make Likelihood histogram
    ml_counts, ml_bins = histogram(ml_sample, bins=50)
    ml_counts, ml_bins = AggregateEmptyBins(ml_counts, ml_bins)
    
    return ml_counts, ml_bins, ml_sample 


#def CalcR2(test, label, random_lab):
def CalcR2(test, random_lab):

    # initialize
#    if random_lab is None:
#        AreTheyIdentical = True
#        while AreTheyIdentical == True:
#            random_lab_ind = np.random.randint(0, len(label))
#            random_lab = label[random_lab_ind]
#        
#            AreTheyIdentical = all(test == random_lab)
#
    r2 = sk.metrics.r2_score(random_lab, test)
    return r2

def MeanR2(test, label):
    #mean_r2 = ft.partial(CalcR2, test=test)
    #mean_r2 = lambda label_i: CalcR2(test, label, label_i)
    mean_r2 = lambda label_i: CalcR2(test, label_i)
    mean_r2_of_test = map(mean_r2, label)
    mean_r2_of_test = sum(mean_r2_of_test) / len(label)
    return mean_r2_of_test







ml_counts, ml_bins, ml_sample = SampleML(label)
greater_than_prob = lambda ml_of_test: np.sum(ml_sample > ml_of_test) / len(ml_sample)


# Example
test_eg, label_eg = PullOut(label, test_ind=None)
ml_of_eg = CalcML(label_eg, test=test_eg)
ml_of_eg = round(ml_of_eg, 5)

prob_of_eg = CountProb(ml_of_eg, ml_bins, ml_counts)
prob_of_eg = round(prob_of_eg, 5)

r2_of_eg = MeanR2(test_eg, label)
print(f"Lk of Eg  : {ml_of_eg}.",
      f"\tBin Prob: {prob_of_eg}",
      f"\tProb of Greater: {greater_than_prob(ml_of_eg)}",
      f"\tR2: {r2_of_eg}"
)


# Metrics of Prediction
ml_of_pred = CalcML(label, test=prediction)
ml_of_pred = round(ml_of_pred, 5)
prob_of_pred = CountProb(ml_of_pred, ml_bins, ml_counts)
prob_of_pred = round(prob_of_pred, 5)

r2_of_pred = MeanR2(prediction, label)
print(f"Lk of Pred: {ml_of_pred}.",
      f"\tBin Prob: {prob_of_pred}.",
      f"\tProb of Greater: {greater_than_prob(ml_of_pred)}",
      f"\tR2: {r2_of_pred}"
)


# Compare mean fo simulations
mean_test = np.mean(label, axis=0)

ml_of_mean = CalcML(label, test=mean_test)
ml_of_mean = round(ml_of_mean, 5)

prob_of_mean = CountProb(ml_of_mean, ml_bins, ml_counts)
prob_of_mean = round(prob_of_mean, 5)

r2_of_mean = MeanR2(mean_test, label)
print(f"Lk of Mean: {ml_of_mean}.",
        f"\tBin Prob: {prob_of_mean}.",
        f"\tProb of Greater: {greater_than_prob(ml_of_mean)}"
        f"\tR2: {r2_of_mean}"
)

# Compare mean of metrics
mean_of_ml = np.mean(ml_sample)
mean_of_ml = round(mean_of_ml, 5)

prob_of_ml = CountProb(mean_of_ml, ml_bins, ml_counts)
prob_of_ml = round(prob_of_ml, 5)


def CalcMetrics(test_ind):
    test_eg, label_eg = PullOut(label, test_ind=test_ind)
    ml_of_eg = CalcML(label_eg, test=test_eg)

    r2_of_eg = MeanR2(test_eg, label_eg)
    return [ml_of_eg, r2_of_eg]

lbog = deepcopy(label)
ml_r2 = map(CalcMetrics, range(len(label)))
ml_r2 = list(ml_r2)
ml_r2 = np.array(ml_r2)

mean_ml2 = np.mean(ml_r2[:,0])
mean_r2 = np.mean(ml_r2[:,1])


print(f"Mean of Lk: {mean_of_ml}.",
      f"\tBin Prob: {prob_of_ml}",
      f"\tProb of Greater: {greater_than_prob(mean_of_ml)}",
        f"\tR2: {mean_r2}"
)





# Highest Density
ml_densities = np.max(ml_counts/np.sum(ml_counts)) / np.diff(ml_bins)
highest_point = np.max(ml_densities) * 1.05

plt.close()

fig = plt.figure()
plt.title("Log Likelihood Histogram")
plt.xlabel("Bootstrapped Log Likelihood of Observation")
#plt.ylabel("Counts")
plt.ylabel("Probability Density")

#ml_counts, ml_bins, plt_bar_thing = plt.hist(ml_sample, bins=20)
plt.plot([ml_of_pred, ml_of_pred],[0,highest_point])#,'r','-')
plt.plot([ml_of_mean, ml_of_mean],[0,highest_point])#,'k','-')
plt.plot([ml_of_eg, ml_of_eg],[0,highest_point])#,'-')#,'lime','-')

#handles, labels = ax.get_legend_handles_labels()
plt.legend(#handles, #labels)
       # [line1, line2, line3],
           ["Log Likelihood of Prediction",
            "Log Likelihood of Mean",
            "Log Likelihood of Example"])

plt.ylim([0, highest_point])

percent_buffer = 10
percent_buffer /= 100
#xlim_low = np.min([ml_of_pred, ml_of_mean, ml_of_eg, *ml_bins])*(1-percent_buffer)
#xlim_high = np.max([ml_of_pred, ml_of_mean, ml_of_eg, *ml_bins])*(1+percent_buffer)

#plt.xlim([xlim_low, xlim_high])
plt.hist(ml_sample, density=True, bins=ml_bins, color='k')
plt.show()





max_ml_ind = np.argmax(ml_sample)

from ProcessCSV import CompareImages
def CompareTwo():
    A = mean_test.reshape((8,8))
    A_title="mean"
    B = label[max_ml_ind].reshape((8,8))
    B_title="Example Label"
    CompareImages(A,B,A_title=A_title, B_title=B_title)

#CompareTwo()
