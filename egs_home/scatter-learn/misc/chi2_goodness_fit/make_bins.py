import numpy as np
import sys
import itertools
import csv

#row=sys.argv[1]
def ProcessRow(row,d=2):
    #row = row.split(",")
    inds = row[1:3]
    inds = map(int, inds)
    inds = list(inds)


    row = row[3:]
    row = list(map(float, row))

    # histograms work liek: b_0 <= x < b_1
    #                       ...
    #                       b_n-1 <= x < b_n
    #                       b_n >= x

    # unconsidered binning...
    row = np.mean(row)
    #row = map(list, row)
    #row = list(row)
    #row = zip(*row)

    #row = itertools.chain(inds,row)
    #row = list(row)
    #row = inds + [row]
    return row

with open("col_3_bk.csv",'r') as fil:
    data_og = fil.read()
data_og = data_og.split("\n")[:-1]

csv_reader = csv.reader(data_og)

#data = [[int(data_i[0],
#         int(data_i[1]),
#         ProcessRow(data_i)] for data_i in csv_reader]
#data = list(data)

data = []
XX = []
FF = []
YY = []
ZZ = []
import matplotlib.pyplot as plt
for data_i in csv_reader:
    l = [int(data_i[0]),
         int(data_i[1]),
        ProcessRow(data_i)
    ]
    data.append(l)

    FF.append(l) 
    XX.append(l[0])
    YY.append(l[1])
    ZZ.append(l[-1])


def Make2D(data):
    dn = np.array(data)
    y_set = np.unique(np.array(data)[:,1])
    data_2d = []
    for y_i in y_set:
        data_y_i = filter(lambda data_j: data_j[1] == y_i, data)
        data_y_i = list(data_y_i)
        data_2d.append(data_y_i)

    data_2d = np.array(data_2d)
    data_2d = data_2d[:,:,-1]
    return data_2d


1/0
def GetPointProb(point, hist):
    # bins are assumed to be sorted
    counts = hist[0]
    bins = hist[1]

    #N_bins = len(bins)
    for bin_ind, bin_j in enumerate(bins):
        if point <= bin_j:
            break
    
    # Probability of being in that bin
    N_in_bin_i = counts[bin_ind]
    PointProb = N_in_bin_i / sum(counts)


    try:
        # Assert that the relevant range is ~backward facing.
        assert bin_ind - 1 >= 0
        return (bins[bin_ind -1], bin_j), PointProb

    # if not backwacks facing.. forwards facing.
    except AssertionError:
        return PointProb, (bin_j, bins[bin_ind + 1])






