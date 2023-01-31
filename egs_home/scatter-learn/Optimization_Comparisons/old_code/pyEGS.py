from CollectSingleResults import CollectSingleResults
import ProcessCSV
from ProcessLabelData import ProcessLabelData
import subprocess


def RunSimulation(rand_num, it_turn):
    #command = f"./sweep_snapshot.sh {rand_num} {it_turn}"
    command = f"./sweep_noclean.sh {rand_num}"# {it_turn}"
    simulation_output = subprocess.run([command],
                                        capture_output=True,
                                        shell=True)
    return simulation_output.stdout

    
def CleanupSimulation(rand_num, run_id):
    subprocess.run([f"./sweep_clean.sh \"{rand_num} {run_id}\""], shell=True)


def ProcessSimulationID(stdout):
    stdout = stdout.decode("utf-8")
    stdout = stdout.split("\n")
    stdout = stdout[-2]
    stdout = stdout.split()
    return stdout


class EGSnrcCall():
    def __init__(lv_min, lv_max):

    def GetOutput(design, it_turn, lv_min, lv_minmax):
        ''' Effectively: _Simulate_'''
        rand_num = ProcessCSV.DesignToRandomNumber(design, base=4, npad=2)
        simulation_stdout = RunSimulation(rand_num, it_turn)
        rand_num, run_id = ProcessSimulationID(simulation_stdout)
    
        sc, dr = CollectSingleResults(rand_num, id_full=run_id)
        prediction = ProcessLabelData(dr, lv_min=lv_min, lv_minmax=lv_minmax)
        prediction = prediction.flatten()
    
        CleanupSimulation(rand_num, run_id)
        return prediction

