import numpy as np
import matplotlib.pyplot as plt

from ProcessInput import ProcessInput
from ProcessLabelData import ProcessLabelData
from TrainMod import SplitTestTrain
from LoadModel import LoadModel, InferNPad
from PyEGSnrc import EGSnrcCall 
from SimulatedAnnealer import *


# Load model
mod = LoadModel()
npad = InferNPad(mod)
f = npad+1

# Get Data
try:
    import sys
    rn = sys.argv[1]

except IndexError:
    # Toy goal
    from CollectSingleResults import CollectSingleResults
    rand_num = 432594882729282400242400315310082#0
    sid, dc = CollectSingleResults(rand_num=rand_num)
    
    # Most recent
#   from LoadRecentData import LoadRecentData
#   dc, sid, mats = LoadRecentData()


# Process Labels for goal
lv, lv_min, lv_minmax = ProcessLabelData(dc)
ssc, sid = ProcessInput(sid, npad)

# Get parameters
N = lv.shape[0]
n = 8
sq = 8


# DEV: THE PRE-TRAINED MODEL WILL NOT HAVE
#      BEEN TRAINED ON THIS TEST/TRAIN 
#      SPLIT
#input_data_train, \
#input_data_test, \
#tri, tei = SplitTestTrain(ssc, N)

# Init Scoper
from ProcessInput import VoxScoper
S = VoxScoper(npad)
scope=lambda s: np.array(S.ScopeObj(s,sq)).reshape((-1,f*f))


# Define goal
ideal_solution = sid[0]
scoped_solution = scope(ideal_solution)#, sq, npad)

# Find prediction on ideal design
goal = mod.predict(scoped_solution)


# Guess Input
material_numbers = np.unique(ideal_solution)
design = np.random.choice(material_numbers,
                          ideal_solution.shape)
# Clear padding
design[:2,:] = 0
design[-2:,:] = 0
design[:,:2] = 0
design[:,-2:] = 0


# For optimization using a surrogate model
Evaluator = SurrogateCall(mod, scope)

# For Bruteforce optimization
#Evaluator = EGSnrcCall(lv_min, lv_max, N_mat, npad)


# Calculate initial score
iter_turn = 0
score_initial = score_func(design,
                           goal,
                           Evaluator,
                           iter_turn)

# Give non-sensical itteration value for ideal.
iter_turn = "inf"
score_ideal = score_func(ideal_solution,
                         goal,
                         Evaluator,
                         iter_turn)
del iter_turn # clear value

print("initial:", score_initial)
print("ideal:", score_ideal)


## OPTIMIZATION ##

# Init optimizer
Opti = SimulatedAnnealer(design,
                         goal,
                         Evaluator,
                         npad,
                         sq,
                         material_numbers)
prob_l = Opti.main()

# Execute operation and return list of probabilities
prob_l = list(prob_l)

print("OPTIMIZATION DONE")
print(f"time taken: {Opti.time_taken}")
##  END OF OPTIMIZATION  ##

##  PLOTTING  ##

# Figure layout
fig, ax = plt.subplot_mosaic([list("abcc"),list("decc")])


# Ideal solution and performance
ideal_solution_scoped = scope(ideal_solution)
ideal_performance = mod.predict(ideal_solution_scoped)
ideal_performance = ideal_performance.reshape((n,n))
ideal_score = score_func(ideal_solution,
                         goal,
                         Evaluator,
                         Opti.iter_turn)
ideal_score = round(ideal_score, 1)

ax['a'].imshow(ideal_solution)
ax['a'].set_title("Ideal Design")#+str(si))
ax['b'].imshow(ideal_performance)
ax['b'].set_title(
         f"Predicted Ideal Performance: MSE = {ideal_score}"
)


# Goal performance
goal_image = goal.reshape((n, n))
ax['c'].imshow(goal_image)
ax['c'].set_title("Simulation Result")


# Best performance
best_solution = Opti.best_solution
ax['d'].imshow(best_solution)
ax['d'].set_title("Best Design")

best_solution_scoped = scope(best_solution)
best_performance = mod.predict(best_solution_scoped)
best_performance = best_performance.reshape((n, n))
best_score = score_func(best_solution,
                        goal,
                        Evaluator,
                        Opti.iter_turn)
best_score = round(best_score, 1)

ax['e'].imshow(best_performance)
ax['e'].set_title(f"Best Performance: MSE = {best_score}")

# Final design and performance
#final_solution = Opti.design
#final_solution_scoped = scope(final_solution)
#final_performance = mod.predict(final_solution_scoped)
#final_performance = final_performance.reshape((n, n))
#final_score = score_func(final_solution, goal, mod)
#final_score = round(final_score, 1)
#
#ax['f'].imshow(Opti.design)
#ax['f'].set_title("Final Design")#+str(s))
#ax['g'].imshow(final_performance)
#ax['g'].set_title(f"Final Performance: MSE = {final_score}")



