class Opti_class():
    def __init__(self, design):
        self.design = design
        self.temp = 1
        self.score_avg = score_initial
        self.score_best = score_initial
        self.score = score_initial
        self.it_turn = 0

        self.cooling_factor = 0.9

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
        self.time_start = time.time()

        s = self.score_avg
        for turn in range(2000):
            #for in_turn in range(n_optim):
            self.it_turn += 1
            
            # Select and save random position's value
            i, j = np.random.randint(npad, npad + sq, 2)
            old_val = self.design[i][j] 
            
            # Mutate design at that random position.
            self.design[i][j] = np.random.choice(material_numbers)
            #other_mats = material_numbers.difference(old_val)
            #new_val = np.random_choice(other_mats)
            #self.design[i][j] = new_val

            # Score new design
            self.score_new = ScoreDesign(self.design, goal, mod, self.it_turn)

            # Update History and running average
            score_hist[self.it_turn % n_optim] = self.score_new
            self.score_avg *= 0.9
            self.score_avg += 0.1*self.score_new
        
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
                    self.score_avg = self.score
        
                else:
                    pass
                
                # Calcuate Probability
                try: 
                    score_ratio = self.score_new / self.score
                    pr = self.temp*np.exp(-(score_ratio/8)**2)
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
        
            print(f"itt: {self.it_turn} | {self.score}")#, self.temp, pr)
            try:
               assert abs(self.score) > 0

            except AssertionError:
                break

        self.time_taken = time.time() - self.time_start

