import subprocess
from CollectSingleResults import CollectSingleResults
from ProcessLabelData import ProcessLabelData


def DesignToRandomNumber(design, base=4, npad=2):
    ''' Converts designs back into random number. \
        Assumes padding'''

    scatter_object = design[npad:-npad, npad:-npad]
    int_in_base = ("".join(map(str,row_i)) for row_i in scattering_object)
    int_in_base = "".join(int_in_base)

    rand_num = int(int_in_base, base)
    return rand_num


def RunSimulation(rand_num, iter_turn):
    ''' Call EGSnrc Simulation, iteration becomes id'''

    terminal_command = [f"./sweep_noclean.sh {rand_num} {iter_turn}"]
    simulation_output = subprocess.run(terminal_command,
                                        capture_output=True,
                                        shell=True)
    return simulation_output.stdout


def ProcessSimulationID(stdout):
    '''Process stdout into simulation ID'''

    stdout = stdout.decode("utf-8")
    stdout = stdout.split("\n")
    stdout = stdout[-2]
    stdout = stdout.split()
    return stdout


def CleanupSimulation(rand_num, run_id):
    '''Move output files of simulation to directory of the most recent simulation'''

    subprocess.run([f"./sweep_clean.sh \"{rand_num} {run_id}\""], shell=True)


class EGSnrcCall():
    def __init__(self, lv_min, lv_minmax, N_mat, npad):
        self.lv_min = lv_min
        self.lv_minmax = lv_minmax

        self.N_mat = N_mat
        self.npad  = npad

    def GetOutput(self, design, iter_turn):
        '''Python function to evaluate a design using EGSnrc simulation. \
           Provides common interface with SurrogatCall.GetOutput()'''
        # Convert design to hex
        rand_num = DesignToRandomNumber(design, base=self.N_mat, npad=self.npad)
    
        # Pass design-hex for EGSnrc simulation
        simulation_stdout = RunSimulation(rand_num, iter_turn)
        rand_num, run_id = ProcessSimulationID(simulation_stdout)
    
        # Load and process simulation results
        sc, dr = CollectSingleResults(rand_num)
        prediction = ProcessLabelData(dr, lv_min=lv_min, lv_minmax=lv_minmax)
        prediction = prediction.flatten()
    
        # Cleanup from simulation
        CleanupSimulation(rand_num, run_id)
        return prediction
    
