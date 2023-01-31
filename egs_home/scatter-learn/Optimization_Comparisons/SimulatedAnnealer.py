import numpy as np
import matplotlib.pyplot as plt
from time import time

def ComparePerformance(score_new, score_old):
    return score_new < score_old


class SurrogateCall():
    def __init__(self, mod, scope):
        self.mod = mod
        self.scope = scope

    def GetOutput(self, design, iter_turn):
        scoped_design = self.scope(design)
        prediction = self.mod.predict(scoped_design)
        return prediction


def loss_func(x, y):
    x -= y
    x **= 2
    return x.sum()


def score_func(design, goal, Evaluator, iter_turn):

    # For optimization with surrogate model
    prediction = Evaluator.GetOutput(design, iter_turn)
    loss = loss_func(prediction, goal)
    
    # Using built-in scoring
    #score = mod.score(scoped_design, goal)

    # To make Lead wall.
    #loss = np.sum(prediction) 

    # Intersting non-determined goal.
    #loss = 1/np.var(prediction) 
    return loss


class SimulatedAnnealer():
    def __init__(self,
                 design, 
                 goal, 
                 Evaluator, 
                 npad, 
                 sq, 
                 material_numbers):
        
        self.npad = npad
        self.sq = sq
        self.material_numbers = material_numbers

        self.goal = goal
        self.Evaluator = Evaluator


        self.temp = 100
        self.cooling_fact = 0.90

        self.iter_turn = 0
        self.design = design

        score_initial = score_func(design,
                                   goal,
                                   self.Evaluator,
                                   self.iter_turn)

        self.score = score_initial
        self.score_avg = score_initial
        self.score_best = score_initial
        
        # Init local score history
        #self.n_optim = 4
        #self.score_hist = np.zeros(self.n_optim)
        
        # Init. figures 
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        #self.SaveFig()

    def SaveFig(self):
        self.ax.imshow(self.design)
        self.ax.set_title(
           f"Optimization Iteration: {self.iter_turn}"
        )
        #self.fig.savefig("".join((
        #  "./optimization_plots/",
        #  f"surrogate/{self.iter_turn}_fig.png"))
        #)
        plt.close()
    
    def main(self):
        self.time_start = time()
        
        # Optimization Loop
        for turn in range(1000):
            self.iter_turn += 1
            
            # Select index to mutate
            i, j = np.random.randint(self.npad,
                                     self.npad + self.sq,
                                     2)

            # Record old value
            old_val = self.design[i][j] 
            
            # Choose new value
            new_val = np.random.choice(self.material_numbers)
            
            # Mutate design
            self.design[i][j] = new_val

            # Score new design
            self.score_new = score_func(self.design,
                                        self.goal,
                                        self.Evaluator,
                                        self.iter_turn)

            # Update History and running average
            #hist_ind = self.iter_turn % n_optim
            #self.score_hist[hist_ind] = self.score_new
            self.score_avg *= 0.9
            self.score_avg += 0.1*self.score_new
        
            ## Optimize
            IsScoreImproved = ComparePerformance(
                                    self.score_new,
                                    self.score
            )
            if IsScoreImproved:
                self.score = self.score_new
                
                # Keep track of best observed design
                IsScoreBest = ComparePerformance(
                                    self.score_new,
                                    self.score_best
                )
                if IsScoreBest:
                    self.score_best = self.score_new
                    self.best_solution = self.design.copy()
        
            else:
                
                # If score increased above the time-weighted
                # average, => cool the optimization.
                if self.score >= self.score_avg:
                    #self.SaveFig()
                    self.temp *= self.cooling_fact
                    self.score_avg = self.score
        
                else:
                    pass
       

                # Accept / reject sampling method
                try: 
                    score_ratio = self.score_new / self.score
                    score_ratio /= 8
                    score_ratio **= 2
                    prob = self.temp*np.exp(-(score_ratio))
                    yield prob

                except ZeroDivisionError:
                    break
                ## lim sp >> s: exp(s/sp) -> 0, pr -> 1
                ## lim sp << s: exp(s/sp) -> -\infty, pr -> 0
                
                u = np.random.uniform()
                if u < prob:
                    # acceptance of `bad' design modification
                    self.score = self.score_new
                    #self.SaveFig()
        
                else:
                    # return to original state.
                    self.design[i][j] = old_val
                    #self.SaveFig()

            print(f"itt: {self.iter_turn} | {self.score}") 
            try:
               assert abs(self.score) > 0

            except AssertionError:
                break

        self.time_taken = time() - self.time_start

