import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py as h5
import time
from LoadRecentData import LoadRecentData
from ProcessInput import ProcessInput, VoxScoper
from ProcessLabelData import ProcessLabelData, NormalizeLabel
from TrainMod import SplitTestTrain
from LoadModel import LoadModel, InferNPad
import pyEGS

from tqdm import tqdm
import random


# Load model
mod = LoadModel()
lv_min = mod.labr_min
lv_minmax = mod.labr_minmax


#mod = None
#lv_min = 0
#lv_minmax = 1


#npad = InferNPad(mod)
npad = 2
f = npad+1


# Get Data
#try:
#    import sys
#    rn = sys.argv[1]

# Get parameters
n = 8
sq = 8
np.random.seed(int(time.time()))
from CollectSingleResults import CollectSingleResults
rand_num = 432594882729282400242400315310080
sid, dc = CollectSingleResults(rand_num=rand_num)
#except IndexError:
#    dc, sid, mats = LoadRecentData()
sid_og = sid.copy()

#lv, lv_min, lv_minmax = NormalizeLabel(dc[:,:,3])
#lv, lv_min, lv_minmax = ProcessLabelData(dc)
lv = ProcessLabelData(dc, lv_min=lv_min, lv_minmax=lv_minmax)


ssc, sid = ProcessInput(sid, npad)
#ideal_solution = sid[0]#.reshape((-1, f*f))
ideal_solution = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 0],
                           [0, 0, 0, 3, 3, 0, 0, 3, 0, 3, 0, 0],
                           [0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0],
                           [0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0],
                           [0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 0, 0],
                           [0, 0, 0, 3, 3, 0, 0, 3, 0, 3, 0, 0],
                           [0, 0, 3, 0, 0, 3, 3, 0, 3, 3, 0, 0],
                           [0, 0, 0, 0, 3, 0, 3, 3, 0, 3, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

design = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0, 0],
                   [0, 0, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0],
                   [0, 0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0],
                   [0, 0, 3, 0, 0, 3, 3, 0, 0, 3, 0, 0],
                   [0, 0, 0, 3, 3, 0, 3, 3, 0, 3, 0, 0],
                   [0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                   [0, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 0],
                   [0, 0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#
checkers = ((np.arange(n**2).reshape((n,n))%2).T + (np.arange(n**2).reshape((n,n))%2))%2
c2 = np.zeros_like(design)
c2[npad:-npad, npad:-npad] = checkers
checkers = c2

## DEV: THE PRE-TRAINED MODEL WILL NOT HAVE
##      BEEN TRAINED ON THIS TEST/TRAIN 
##      SPLIT
#input_data_train, input_data_test, tri, tei = SplitTestTrain(ssc, N)

Scoper = VoxScoper(npad)
scope = lambda s: np.array(Scoper.ScopeObj(s, sq)).reshape((-1,f*f))


# Guess Input
material_numbers = np.unique(ideal_solution)
#material_numbers = np.array([0,3])
#design = np.random.choice(material_numbers,
#                          ideal_solution.shape)
material_numbers=set(material_numbers)

material_choices = np.array(sorted(material_numbers))


def RandomDesign():
    rand_design = np.random.randint(0,len(material_numbers), (n,n))
    rand_design = material_choices[rand_design]

    wide_view = np.zeros_like(ideal_solution)
    wide_view[npad:-npad, npad:-npad] = rand_design
    return wide_view


# Assert that padding of design is zero
design[:npad,:] = 0
design[-npad:,:] = 0
design[:,:npad] = 0
design[:,-npad:] = 0

good_guess= [[1,1,0,1,1,0,1,1],
             [1,1,0,1,1,0,1,1],
             [0,0,1,0,0,1,0,0],
             [1,1,0,1,1,0,1,1],
             [1,1,0,1,1,0,1,1],
             [0,0,1,0,0,1,0,0],
             [1,1,0,1,1,0,1,1],
             [1,1,0,1,1,0,1,1]]

gg = np.zeros_like(ideal_solution)
gg[npad:-npad, npad:-npad] = good_guess
good_guess = gg*3



## Simulated annealing
#   - use r2 as loss?


def ComparePerformance(score_new, score_old):
    return score_new <= score_old

def GetSurrogateOutput(design, mod):
    scoped_design = scope(design)
    return mod.predict(scoped_design)

def loss_func(x, y):
    # MSE loss
    x -= y
    x **= 2 
    return np.sum(x)

eval_call = GetSurrogateOutput
#eval_call = pyEGS.GetManualOutput

def ScoreDesign(design, goal, mod, it_turn):
    # eval_call = pyEGS.GetManualOutput
    #prediction = eval_call(design, it_turn, lv_min, lv_minmax)

    # eval_call = GetSurrogateOutput
    #prediction = eval_call(design, mod)

    #loss = loss_func(prediction, goal)


    #score = mod.score(scoped_design, goal)
    #loss = np.sum(prediction) # To make Lead wall.
    #loss = -np.var(design[npad:-npad, npad:-npad]) # Intersting non-determined goal.

    score = scope(design)
    score = np.var(score)#, axis=1)
    #score = np.mean(score)
    loss = 1/score

    #loss = np.var(design[npad:-npad, npad:-npad]) # Intersting non-determined goal.
    return loss

it_turn = 0
goal = pyEGS.GetManualOutput(ideal_solution, -1, lv_min, lv_minmax)
goal_og = goal.copy()

score_initial = ScoreDesign(design, goal, mod, it_turn)

# Give non-sensical itteration value for ideal.
it_turn = "inf"
score_ideal = ScoreDesign(ideal_solution, goal, mod, it_turn)

del it_turn
print("initial:", score_initial)
print("ideal:", score_ideal)

n_optim = 10
score_hist = np.zeros(n_optim)


class Opti_class():
    def __init__(self, design):
        self.design = design
        self.temp = 1
        self.score_avg = score_initial
        self.score_best = score_initial
        self.score = score_initial
        self.it_turn = 0

        self.cooling_factor = 0.9925

        #fig, ax = plt.subplots()
        #self.fig = fig
        #self.ax = ax
        #self.SaveFig()

    #def SaveFig(self):
    #    self.ax.imshow(self.design)
    #    self.ax.set_title(f"Optimization Iteration: {self.it_turn}")
    #    self.fig.savefig(f"./optimization_plots/simulation/py_figs/{self.it_turn}_fig.png")
    #    plt.close()
    
    def main(self):
        pr = 1
        self.time_start = time.time()

        s = self.score_avg
        i, j = np.random.randint(npad, npad + sq, 2)
        #for turn in tqdm(range(250000)):
        for turn in range(5000000):
            #for in_turn in range(n_optim):
            self.it_turn += 1
            
            # Select and save random position's value
            #i_old, j_old = i, j
            #while (i_old == i) or (j_old == j):
            i, j = np.random.randint(npad, npad + sq, 2)

            old_val = self.design[i][j] 
            
            # Mutate design at that random position.
            #self.design[i][j] = np.random.choice(material_numbers)
            #new_val =  np.random.choice(material_numbers)

            other_mats = material_numbers.difference({old_val})
            #other_mats = list(other_mats)
            #new_val = np.random.choice(other_mats)
            new_val = random.sample(other_mats, k=1)[0]
            #assert int(new_val) != int(old_val)
            self.design[i][j] = new_val

            # Score new design
            self.score_new = ScoreDesign(self.design, goal, mod, self.it_turn)

            # Update History and running average
            #self.score_avg -= 1/n_optim * score_hist[self.it_turn % n_optim]
            score_hist[self.it_turn % n_optim] = self.score_new
            self.score_avg = np.mean(score_hist)

            #self.score_avg *= 0.9
            #self.score_avg += 0.1*self.score_new
        
            ## Optimize
            IsScoreImproved = ComparePerformance(self.score_new, self.score)
            if IsScoreImproved:
                self.score = self.score_new
                
                # If score has improved, it could be best-observed design
                IsScoreBest = ComparePerformance(self.score_new, self.score_best)
                if IsScoreBest:
                    self.score_best = self.score_new
                    self.best_solution = self.design.copy()


            # Score not improved. Probabilistic acceptance. 
            else:
                # If current score worse than average, cool optimization. 
                IsTrendingGood = ComparePerformance(self.score, self.score_avg)  
                if not IsTrendingGood:
                    #self.SaveFig()

                    # Cool temperature
                    self.temp *= self.cooling_factor

                    # Reset running avg.
                    self.score_avg = np.mean(score_hist)

                    #self.score_avg = self.score
        
                else:
                    pass
                
                # Calcuate Probability
                try: 
                    score_ratio = self.score_new / self.score
                    pr = self.temp*np.exp(-(score_ratio/8)**2)
                    #pr = min(0.75, pr)
                    yield pr
                
                # In minimization, score could reach zero
                except ZeroDivisionError:
                    break
                ## lim sp >> s: exp(s/sp) -> 0, pr -> 1
                ## lim sp << s: exp(s/sp) -> -\infty, pr -> 0


                # Probabilistic Acceptance
                u = np.random.uniform()
                if u < pr:
                    self.score = self.score_new
                    #self.SaveFig()

                # Probabilistic Rejection 
                else:
                    # return to original state.
                    self.design[i][j] = old_val
        
            print(f"itt: {self.it_turn} | {round(self.score, 3)} | {round(pr,3)}")#, self.temp, pr)
            #try:
            #   assert abs(self.score) > 0

            #except AssertionError:
            #    break

        self.time_taken = time.time() - self.time_start

Opti = Opti_class(design)
probability_history = Opti.main()

# Lazy evaluation here.
probability_history = list(probability_history)
score_final = ScoreDesign(Opti.design, goal, mod, Opti.it_turn)

#final_design_scoped = scope(Opti.design)
#final_performance = mod.predict(final_design_scoped)
#final_performance = final_performance.reshape((n, n))
#final_score = ScoreDesign(Opti.design, goal, mod)

# Goal performance
goal_image = goal.reshape((n, n))

# Ideal solution and performance
#ideal_solution_scoped = scope(ideal_solution)
#ideal_performance = mod.predict(ideal_solution_scoped)

ideal_performance = eval_call(ideal_solution, mod)
#ideal_performance = eval_call(ideal_solution, "ideal", lv_min, lv_minmax)
ideal_performance = ideal_performance.reshape((n,n))
ideal_score = ScoreDesign(ideal_solution, goal, mod, Opti.it_turn)
#ideal_score = round(ideal_score, 1)


# Best solution and performance
best_solution = Opti.best_solution
#best_performance = eval_call(best_solution, "best", lv_min, lv_minmax)
best_performance = eval_call(best_solution, mod)
best_performance = best_performance.reshape((n,n))

#best_performance = mod.predict(best_solution_scoped)
#best_performance = best_performance.reshape((n, n))
best_score = ScoreDesign(best_solution, goal, mod, Opti.it_turn)
#best_score = round(best_score, 1)

# Common colourbar for predict/ideal performance and goal
from ProcessCSV import MakeCommonColorbar
performances = np.dstack((ideal_performance, best_performance))
per_norm, per_cmap, per_mappable = MakeCommonColorbar(goal_image, performances)

fig, ax = plt.subplot_mosaic([list("abee"),list("cdee")], figsize=(16,8))#,list("fgcc")])

from ColorbarFuncs import MakeMaterialsColorbar
design_norm, design_cmap, design_mappable, design_ticks = MakeMaterialsColorbar(
                                                            ideal_solution,
                                                            best_solution,
                                                            fig)


# Ideal solution and performance
ax['a'].imshow(ideal_solution, norm=design_norm, cmap=design_cmap)
ax['a'].set_title("Ideal Design")#+str(si))
ax['b'].imshow(ideal_performance, norm=per_norm, cmap=per_cmap)
ax['b'].set_title(f"Predicted Ideal Performance: MSE = {round(ideal_score, 1)}")


ax['c'].imshow(best_solution, norm=design_norm, cmap=design_cmap)
ax['c'].set_title("Best Design")


ax['d'].imshow(best_performance, norm=per_norm, cmap=per_cmap)
ax['d'].set_title(f"Best Performance: MSE = {round(best_score, 1)}")

# Goal performance
ime = ax['e'].imshow(goal_image, norm=per_norm, cmap=per_cmap)
ax['e'].set_title("Simulation Result")


# Final design and performance
#final_solution = Opti.design
#final_solution_scoped = scope(final_solution)
#final_performance = mod.predict(final_solution_scoped)
#final_performance = final_performance.reshape((n, n))
#final_score = ScoreDesign(final_solution, goal, mod)
#final_score = round(final_score, 1)
#
#ax['f'].imshow(Opti.design)
#ax['f'].set_title("Final Design")#+str(s))
#ax['g'].imshow(final_performance)
#ax['g'].set_title(f"Final Performance: MSE = {final_score}")



## Make Colorbar.
# .get_position() gives Bbox w/ [[xmin, ymin],[xmax,ymax]]
per_axis_pos = ax['e'].get_position()
per_axis_pos = np.array(per_axis_pos)
per_axis_left = np.min(per_axis_pos[:,0])
per_axis_right = np.max(per_axis_pos[:,0])
per_cb_left = per_axis_right * 1.01
per_cb_width = 0.025

per_axis_bottom = np.min(per_axis_pos[:,1])
per_axis_top = np.max(per_axis_pos[:,1])
per_cb_height = per_axis_top - per_axis_bottom
per_cb_bottom = per_axis_bottom
#per_axis_height += per_axis_bottom

# plt.axes takes f-tuple of floats *rect* = ``[[left, bottom, width, height]]
per_cb_axis = plt.axes([per_cb_left, per_cb_bottom, per_cb_width, per_cb_height])

# Predictions are going to be lower, since it is taking an average...
#   > There will be on voxel that only has maximum values of irradiation.
per_cb = plt.colorbar(per_mappable, cax=per_cb_axis, orientation='vertical')
per_cb.set_label("Normalize Absorbed Dose")


## Label plot
optimization_type = str(eval_call)
optimization_type = optimization_type.split()[1]
if optimization_type == "GetSurrogateOutput":
    fig.suptitle("Design Optimization with Surrogate Model Evalutation")
    plot_name = "optimization_results_-_surrogate.png"
elif optimization_type == "GetManualOutput":
    fig.suptitle("Design Optimization with EGSnrc Simulation Evalutation")
    plot_name = "optimization_results_-_simulation.png"

#fig.savefig(plot_name)
fig.show()

print(f"time taken: {Opti.time_taken}")


