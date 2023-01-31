import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import functools as ft

import sys
from ProcessCSV import FullProcessSingle
from ProcessLabelData import ProcessDistData

np.random.seed(123345)

#mu = 0
#std = 1
N = 1000
n_means = 5000

try:
    rn = sys.argv[1]

except IndexError:
    rn = "62783361878388904048055595911667886675"
    #rn = "252785838956249757503353786711590552658"
#    rn = "334800398373121594589956989"

input_object_path = "".join((
                    f"./{rn}_-_detector_results/",
                    f"scatter-learn_-_num_0-{rn}_-_scatter_input.csv"))

from LoadModel import LoadModel, InferNPad, GetExpandedShape
mod = LoadModel()
lr_min = mod.lr_min
lr_minmax = mod.lr_minmax

# Calculate npad from model
npad = InferNPad(mod)

# Get test object input.
input_data = FullProcessSingle(input_object_path, npad)
prediction = mod.predict(input_data)

distribution_path = "".join((
                    f"./{rn}_-_detector_results/",
                    "column_3.csv"))




def Chi2_i(prediction_i, label_dist, population_variance=None, population_mean=None):
    #Cowan p. 104
    # The quantity (y_i - f(x_i, w)) / s_i is a measure of deviation between the
    # ith measurement y_i and the function f(s_i, w), so Chi^2 is a measure of
    # the total agreement between ovserved data and hypothesis. It can be shown
    # that if

    # 1) y_i, i=1,...,N are independent Gaussian random variables [...]
    #       > mean of distribution and expectation of the mean are expected to be
    #         the same.
    #       > !!! wikipedia says this should be calculated about a sample mean
    #             (not population mean)
    if population_mean is None:
        y_i = np.mean(label_dist)
    else:
        y_i = population_mean

    # [...] with known variances s_i^2
    #s_i = np.var(label_dist)
    if population_variance is None:
        population_variance = np.var(label_dist)

    population_variance_of_the_mean = population_variance / len(label_dist)

    chi2_i = (y_i - prediction_i)**2 / population_variance_of_the_mean
    return chi2_i


def GetSample(u, std, N):
    term = np.sqrt(3)*std
    low = u - term
    high = u + term
    return np.random.uniform(low=low, high=high, size=N)

def GetX():
    mu_max = 1000
    mu = np.random.randint(0, mu_max)
    std = np.random.uniform() * mu_max

    sample = GetSample(mu, std, N)
    pred = np.random.normal(loc=mu, scale=std/N, size=1).item()
    chi2 = Chi2_i(pred, sample, population_variance=std**2)
    #sample = np.random.normal(loc=mu, scale=std, size=N)
    #pred = np.random.normal(loc=mu, scale=std, size=1).item()
    #chi2 = Chi2_i(pred, None, population_variance=std**2, population_mean=mu)

    #, std/np.sqrt(N)) >> already taken down inside chi_2

    #chi2 /= N-1
    

    return chi2


from ScaleBins import StandardCountBins, StandardWidthBins


c2_demo = [GetX() for test_i in range(n_means)]
c2_demo = np.array(c2_demo) #np.fromiter(c2_demo, dtype=float)
bins, expected_counts = StandardCountBins(c2_demo)
#bins, expected_counts = StandardWidthBins(c2_demo)

print("plotting")

plt.close()
fig = plt.figure()
actual_counts, dummy_bins = np.histogram(c2_demo, bins=bins)
plt_counts, plt_bins, bar_container = plt.hist(c2_demo, density=True, bins=bins)


df = 1
c2_ls = np.linspace(np.min(c2_demo), np.max(c2_demo), 1000)
c2_prob = sts.chi2.pdf(c2_ls, df)#, scale=std)
plt.plot(c2_ls, c2_prob)

#plt.ylim([0, np.mean(plt_counts[:2])])
plt.ylim([0, np.max(plt_counts)])
fig.show()


## DOUBLE CHI^2 :::: USE PEARSON'S CHI^2 TO TEST IF DISTRIBUTION IS CHI^2 DISTRIBUTION.
#ep = [sts.chi2.cdf(bins[i+1],df=df) - sts.chi2.cdf(bins[i],df=df) for i in range(len(bins)-1)]

#expected_counts = np.array(ep)
#expected_counts *= np.sum(np_counts)

#expected_count = map(c2_pd, bins)
#= list(expected_count)
df_double = len(bins)
double_chi2 = zip(actual_counts, expected_counts)
double_chi2 = ((obs - ep)**2/ep for obs, ep in double_chi2)

#double_chi2 = ((counts[ind],sts.chi2.pdf(bins[ind])) for ind in range(len(counts)))
double_chi2 = sum(double_chi2)
prob_2c2 = sts.chi2.pdf(double_chi2, df=df_double)
cprob_2c2 = sts.chi2.cdf(np.infty, df=df_double) - sts.chi2.cdf(double_chi2, df=df_double)
#cprob_2c2 = cprob_2c2
print(f"df_double: {df_double}")
print(f"2-chi2: {double_chi2}")
print(f"chi^2/df : {double_chi2/df_double}")
print("")
print(f"prob: {prob_2c2}")
print(f"P-value: {cprob_2c2}")

#plt.figure()
#plt.hist(double_chi2)
#plt.show()
