import numpy as np
from scipy import stats as sts
import time
import matplotlib.pyplot as plt
np.random.seed(int(time.time()))


def sample_test():
    u = np.random.randint(0,1000)
    s = 10 

    #u = 0
    #s = 1 #2.5
    n = 21
    #sample = np.random.normal(loc=u, scale=s, size=n)
    sample = np.random.normal(loc=u-s/2, scale=u+s/2, size=n)
    
    df = n - 1
    c2 = c2_stat(sample, s)
    return c2, df

def sample_prob():
    c2, df = sample_test()
    prob = c2_prob(c2, df)
    return prob

def many_test():
    mt = (sample_prob() >= 0.95 for test_i in range(10000))
    ratio = np.unique(np.fromiter(mt, dtype=bool), return_counts=True)
    return ratio

def c2_stat(sample, nominal_var):
    sample_mean = np.mean(sample)
    
    # statistic `T'
    c2 = sample - sample_mean
    c2 **= 2
    c2 = np.sum(c2)
    c2 /= nominal_var

    return c2

def c2_prob(c2, df):
    return sts.chi2.cdf(c2, df)

#success_vs_loss, counts = many_test()
#mt = many_test()
#print(mt)


c2_demo = (sample_test()[0] for test_i in range(1000))
c2_demo = np.fromiter(c2_demo, dtype=float)

fig = plt.figure()
plt.hist(c2_demo, density=True)

df = 20
c2_ls = np.linspace(np.min(c2_demo), np.max(c2_demo), 1000)
c2_prob = sts.chi2.pdf(c2_ls, df, loc=0, scale=10)
plt.plot(c2_ls, c2_prob)
plt.show()
