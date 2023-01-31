import sys
from ProcessLabelData import ProcessLabelData
from ProcessCSV import ProcessInputCSV
from ProcessInput import ProcessInput
import subprocess

def CollectSingleResults(rand_num=None, id_full=0):
#if True:
    if rand_num is None:
        rand_num = sys.argv[1]
    #rn = 432594882729282400242400315310082#0
   
    # Make path variables
    if str(id_full).__contains__(str(rand_num)):
        scatter_object_path = f"scatter-learn_-_{id_full}_-_scatter_input.csv"
        detector_results_path = f"scatter-learn_-_{id_full}_-_detector_results.csv"

    else:
        scatter_object_path = f"scatter-learn_-_num_{id_full}-{rand_num}_-_scatter_input.csv"
        detector_results_path = f"scatter-learn_-_num_{id_full}-{rand_num}_-_detector_results.csv"


    try:
        # Load results from CSV files
        scatter_object = ProcessInputCSV(scatter_object_path)
        detector_results = ProcessInputCSV(detector_results_path)

    except FileNotFoundError:
        process_params = [f"./sweep_noclean.sh {rand_num}"]
        process = subprocess.run(process_params, shell=True)

        # Load results from CSV files
        scatter_object = ProcessInputCSV(scatter_object_path)
        detector_results = ProcessInputCSV(detector_results_path)
    

    return scatter_object, detector_results

    # Process results
    #scatter_object_scoped, scatter_object = ProcessInput(scatter_object, 3)
    #detector_results = ProcessLabelData(detector_results)



#if __name__ == "__main__":
#    so, dr = 

#ssc, dc = CollectSingleResults()

    

