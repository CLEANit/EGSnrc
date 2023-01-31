import numpy as np
from scipy import stats as sts

def _CalcExpectedCounts(xmin, xmax, df, N_data):
    cdf_max = sts.chi2.cdf(xmax, df=df)
    cdf_min = sts.chi2.cdf(xmin, df=df)

    expected_counts = N_data * (cdf_max - cdf_min)

    return expected_counts

def StandardCountBins(x_og, CalcExpectedCounts=_CalcExpectedCounts):
    '''Construct bins so that there will always be a minimum of N items in each'''
    x = x_og.copy()
    x.sort()
    N_data = len(x)
    df = 1 #N_data

    bins = [max(x)]
    expected_counts = []

    # Do
    bin_edge = bins[-1]
    diff_bin_max = np.argsort(np.abs(x - bin_edge))
    #diff_bin_max = diff_bin_max[::-1]
    
    # First bin reaches all the way to infinity.
    bin_edge = np.infty

    
   
    try:
        while min(bins) > min(x):
            #df = min(len(bins) - 1, 1)
            #df = len(bins)
            # Start at back of bins array

            expected_count_i = 0
            min_edge_ind = 0
            while expected_count_i < 50:
                min_edge_ind += 1
                bin_guess = x[diff_bin_max[min_edge_ind]]
                expected_count_i = CalcExpectedCounts(bin_guess,
                                                     bin_edge,
                                                     df,
                                                     N_data)

            bins.append(bin_guess)
            expected_counts.append(expected_count_i)

            # Do
            bin_edge = bins[-1]
            diff_bin_max = np.argsort(x - bin_edge)
            diff_bin_max = diff_bin_max[::-1]

    except IndexError:
        # if it is over run, then throw the rest in the last bin.
        bins[-1] = 0 #min(x)
        expected_count_i = CalcExpectedCounts(bins[-1],
                                              bins[-2],
                                              df,
                                              N_data)
        expected_counts[-1] = expected_count_i


    


    bins = bins[::-1]
    df = 1 

    ep = [sts.chi2.cdf(bins[i+1],df=df) - sts.chi2.cdf(bins[i],df=df) for i in range(len(bins)-1)]
    expected_counts = np.array(ep)
    expected_counts *= len(x)


    #expected_counts = np.array(expected_counts)
    #expected_counts = expected_counts[::-1]
    #expected_counts /= np.sum(expected_counts)
    return bins, expected_counts


def StandardWidthBins(c2_demo):    
    less_bins = 0
    np_counts = [0]
    print(f"type(c2_demo): {type(c2_demo)}")
    n_data = len(c2_demo)
    while np.min(np_counts) < 5:
        N_bin = n_data//10 - less_bins
        #counts, bins, bar_container = plt.hist(c2_demo, density=True, bins=N_bin)
        np_counts, np_bins = np.histogram(c2_demo, bins=N_bin)
        less_bins += 1

    df = N_bin 

    ep = [sts.chi2.cdf(np_bins[i+1],df=df) - sts.chi2.cdf(np_bins[i],df=df) for i in range(len(np_bins)-1)]

    expected_counts = np.array(ep)
    expected_counts *= np.sum(np_counts)



    return np_bins, expected_counts

