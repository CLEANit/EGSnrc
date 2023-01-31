import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import functools as ft

#mu = 0
#std = 1
N = 10000
n_means = 2000



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


def GetSample(mu, std, N):
    term = np.sqrt(3)*std
    low = mu - term
    high = mu + term
    return np.random.uniform(low=low, high=high, size=N)
    #return np.random.normal(loc=mu, scale=std, size=N)

def GetX():
    mu_max = 1000
    mu = np.random.randint(0, mu_max)
    std = np.random.uniform() * mu_max * 2
    #s = s
    #N = N
    sample = GetSample(mu, std, N)
    # The prediction must be from the sample // Chi^2 only works if value is gaussian
    pred = mu #GetSample(mu, std, 1)

    # If the prediction were to be from the distribution of the mean
    #pred = np.random.normal(loc=mu, scale=std/np.sqrt(N))

    #pred = GetSample(mu, std/n, 1)

    chi2 = Chi2_i(pred, sample, std**2)
    #, std/np.sqrt(N)) >> already taken down inside chi_2

    #chi2 /= N-1
    

    return chi2

c2_demo = (GetX() for test_i in range(n_means))
c2_demo = np.fromiter(c2_demo, dtype=float)

print("plotting")
less_bins = 0
np_counts = [0]
while np.min(np_counts) < 1:
    N_bin = n_means//10 - less_bins
    #counts, bins, bar_container = plt.hist(c2_demo, density=True, bins=N_bin)
    np_counts, np_bins = np.histogram(c2_demo, bins=N_bin)
    less_bins += 1

plt.close()
fig = plt.figure()
counts, bins, bar_container = plt.hist(c2_demo, density=True, bins=np_bins)


df = 1
c2_ls = np.linspace(np.min(c2_demo), np.max(c2_demo), 1000)
c2_prob = sts.chi2.pdf(c2_ls, df)#, scale=std)
plt.plot(c2_ls, c2_prob)

plt.ylim([0, np.mean(counts[:2])])
fig.show()

#frac_before_one = np.sum(counts[np.where(bins <= 1)[0]]) / np.sum(counts)

## DOUBLE CHI^2 :::: USE PEARSON'S CHI^2 TO TEST IF DISTRIBUTION IS CHI^2 DISTRIBUTION.
ep = [sts.chi2.cdf(bins[i+1],df=df) - sts.chi2.cdf(bins[i],df=df) for i in range(len(bins)-1)]

expected_counts = np.array(ep)
expected_counts *= np.sum(np_counts)

#expected_count = map(c2_pd, bins)
#= list(expected_count)
double_chi2 = zip(np_counts, expected_counts)
double_chi2 = ((obs - ep)**2/ep for obs, ep in double_chi2)

#double_chi2 = ((counts[ind],sts.chi2.pdf(bins[ind])) for ind in range(len(counts)))
double_chi2 = sum(double_chi2)
prob_2c2 = sts.chi2.pdf(double_chi2, df=N_bin)
cprob_2c2 = sts.chi2.cdf(np.infty, df=N_bin) - sts.chi2.cdf(double_chi2, df=N_bin)
#cprob_2c2 = cprob_2c2
print(f"N_bin: {N_bin}")
print(f"2-chi2: {double_chi2}")
print(f"chi^2/df : {double_chi2/N_bin}")
print("")
print(f"prob: {prob_2c2}")
print(f"P-value: {cprob_2c2}")

#plt.figure()
#plt.hist(double_chi2)
#plt.show()
