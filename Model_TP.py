class TPLearner():

    def __init__(self, values):
        self.name = "tp"
        self.values = values
        
        # Parameters of Dirichlet distribution (pseudocounts)
        self.alphas = [[1 for _ in values] for _ in values]

        # Memory of the agent (only remembering the previous)
        self.previous = ""


    def reset(self):
        '''
            Resets all learning so far.
        '''
        self.previous = ""
        self.alphas = [[1 for _ in self.values] for _ in self.values]


    def process_observation(self, obs):
        '''
            Processes the seen shape such per the model (i.e. chunking, TP etc)
        '''
        index = self.values.index(obs)

        # If no shape has been seen yet, distribute 'weight'
        # of observation across all possible values uniformly
        if self.previous == "":
            for row in self.alphas:
                row[index] += (1/len(self.values))

        # Else just update the likelihood of the observed shape
        # given the previous shape in memory
        else:
            prev_index = self.values.index(self.previous)
            self.alphas[prev_index][index] += 1

        self.previous = obs


    def get_probabilities(self):
        '''
            Get the next prediction. This is automatically updated given
            a series of shapes. I.e. no argument is needed for this method.
        '''
        if self.previous == "":
            return [1.0/len(self.values) for _ in self.values]
        else:
            value = self.previous
            value_index = self.values.index(value)
            row = self.alphas[value_index]
            return [a / sum(row) for a in row]


    def get_number_parameters(self):
        '''
            Returns the number of parameters used by the agent.
            This is used to calculate the BIC score
        '''
        nr_parameters = len(self.values)
        for row in self.alphas:
            nr_parameters += len(row)
        return nr_parameters







