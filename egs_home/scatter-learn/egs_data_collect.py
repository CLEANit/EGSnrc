import numpy as np
import pandas as pd
import h5py as h5
import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm


cwd = os.getcwd()
os.chdir("../..")
mats_file = os.listdir("og_results/include_dir")
mats_file = mats_file[0]
with open(f"og_results/include_dir/{mats_file}", 'r') as include_file:
    mats = include_file.readline()

mats = mats.split("=")[1]
mats = mats.strip()
mats = mats.split(", ")

os.chdir(cwd)

with h5.File("croped_data_binary.hdf5",'w') as ddf:

    ddf.create_dataset("mats", data=mats)
#    #ddf.create_dataset("scope_data",data=scope_data)    #Collect scope data
#    #ddf.create_dataset("scope_data_coords",data=scope_data_coords) #Collect scope data
#    detsl = [[int(y[y.index("_")+1:]) for y in x.split("_-_")[1:-1]] for x in fl]
#    #ddf.create_dataset("scope_data_labels",data=detsl)
#    ## Get 
#    #with open(data_name+"_list",'r') as f: #Do once so detsl/data_table isn't created twice
#    #    fl = f.read()
#    #fl = fl.split("\n")[:-1]
#
    #Now collect .csv datas
    bad_inds = []
    for data_name in ["detector_results","scatter_input"][::-1]: #by doing this twice.. how to crop??
        
        # Get names of .csv files of type "data_name"
        #with open(data_name+"_list",'r') as f:
        #    fl = f.read()
        #fl = fl.split("\n")[:-1]
        fl = os.listdir()
        fl = filter(lambda x: x.count(data_name) > 0, fl)
        fl = sorted(fl)


        # For each result, get the random number 
        detsl = []
        for csv_file in fl:
            random_number = csv_file.split("_-_")[1]
            try:
                random_number = random_number[random_number.index("-")+1:]
            except ValueError:
                pass
            detsl.append(random_number)

        df = pd.DataFrame(detsl)
        ddf.create_dataset(data_name+"_random_numbers",data=np.double(df.to_numpy()))
        
        ##^ data table
        deti = pd.read_csv(fl[0],header=None).to_numpy()
        det_data_shape = [len(fl)]+list(deti.shape)
        det_data = np.empty(tuple(det_data_shape))
        #
        det_data[0] = deti
        bad_inds = []
        for i in tqdm(range(1,len(fl))):
            #try:
            det_data[i] = pd.read_csv(fl[i],header=None).to_numpy()
            #except:
            #    bad_inds.append(i)
        #good_inds = list(range(len(fl)))
        #for bad_i in bad_inds:
        #    good_inds.remove(bad_i)
        #print(bad_inds) 
        #good_inds = np.array(good_inds)
        #det_data = det_data[good_inds]
        ddf.create_dataset(data_name,data=det_data)


