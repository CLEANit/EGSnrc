def LoadRecentData(data_path=None, history_index=0):
    import sys
    import os
    import h5py as h5
   
    if data_path is None:
        try:
            data_path = sys.argv[1]
        
        except IndexError:

            history_index *= -1
            history_index -= 1

            ls = os.listdir()
            #ls = filter(lambda x: x.count("run_-_2022-")>0, ls)
            ls = filter(lambda x: x.count("run_-_")>0, ls)
            ls = sorted(ls)
            print(ls)
            data_path = ls[history_index]
        
        # Big data @ "~/egsnrc_runs/run_-_2021-05-04_-_1/croped_data_binary.hdf5"
        #with h5.File("/home/user/egsnrc_runs/run_-_2021-05-04_-_1/croped_data_binary.hdf5",'r
        
    print(f"training on: {data_path}")
    with h5.File(f"{data_path}/croped_data_binary.hdf5",'r') as hf:
        dc = hf["detector_results"][:]#.copy()
        #dct = hf["detector_results_random_numbers"][:].copy()
        sid = hf["scatter_input"][:]#.copy()

        mats = hf["mats"][:]
        try:
            mats_map = map(ConvertBytesToString, mats)
            mats = list(mats_map)

        except AttributeError:
            mats = list(mats)
           #pass 

    ncase = GetNCase(data_path=data_path)
    sigma = GetSigma(data_path=data_path)
    return dc, sid, mats, ncase, sigma

def ConvertBytesToString(b):
    s = b.decode("utf-8")
    return s

def GetNCase(data_path=None):
    import subprocess
    if data_path is None :
        simulation_output = subprocess.run(["grep 'ncase' $(cat most_recent_run.txt)/og_results/egsinp_dir/* | sed 's/.*ncase = //' | sort -u"],
                                           capture_output=True,
                                           shell=True)

    else:
        simulation_output = subprocess.run([f"grep 'ncase' {data_path}/og_results/egsinp_dir/* | sed 's/.*ncase = //' | sort -u"],
                                           capture_output=True,
                                           shell=True)


    ncase = simulation_output.stdout
    ncase = ConvertBytesToString(ncase)
    ncase = int(ncase)
    #print(simulation_output.stdout)
    return ncase

def GetSigma(data_path=None):
    import subprocess
    if data_path is None :
        simulation_output = subprocess.run(["grep 'sigma' $(cat most_recent_run.txt)/og_results/egsinp_dir/* | sed 's/.*sigma = //' | sort -u"],
                                           capture_output=True,
                                           shell=True)

    else:
        simulation_output = subprocess.run([f"grep 'sigma' {data_path}/og_results/egsinp_dir/* | sed 's/.*sigma = //' | sort -u"],
                                           capture_output=True,
                                           shell=True)


    sigma = simulation_output.stdout
    sigma = ConvertBytesToString(sigma)
    sigma = float(sigma)
    #print(simulation_output.stdout)
    return sigma 
    

