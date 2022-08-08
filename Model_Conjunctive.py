class ConjunctiveChunkLearner():

    def __init__(self, values):
        self.name = "conjunctive"

        self.values = values
        self.reset()

        self.chunk_length = 3
        

    def reset(self):
        '''
            Resets all learning so far.
        '''
        self.first_prob = [len(self.values) for _ in self.values]
        self.second_prob = dict()
        for v in self.values:
            self.second_prob[str([v])] = [1 for _ in self.values]

        self.third_prob = dict()
        for v1 in self.values:
            for v2 in self.values:
                self.third_prob[str([v1, v2])] = [1 for _ in self.values]

        self.memory = []


    def process_observation(self, obs):
        '''
            Processes the seen shape such per the model (i.e. chunking, TP etc)
        '''
        index = self.values.index(obs)

        l = len(self.memory)
        if l == 0:
            self.first_prob[index] += 1
        elif l == 1:
            self.second_prob[str(self.memory)][index] += 1
        else:
            self.third_prob[str(self.memory)][index] += 1

        self.memory.append(obs)
        if (len(self.memory) >= 3):
            self.memory = []


    def normalize(self, li):
        '''
            Returns a new list such that the values
            of the argument now sum to 1
        '''
        return [l/sum(li) for l in li]


    def get_probabilities(self):
        '''
            Get the next prediction. This is automatically updated given
            a series of shapes. I.e. no argument is needed for this method.
        '''
        if len(self.memory) == 0:
            return self.normalize(self.first_prob)
    
        elif len(self.memory) == 1:
            return self.normalize(self.second_prob[str(self.memory)])

        else:
            return self.normalize(self.third_prob[str(self.memory)])


    def get_number_parameters(self):
        '''
            Returns the number of parameters used by the agent.
            This is used to calculate the BIC score
        '''
        nr_parameters = 0
        for i in range(self.chunk_length):
            nr_parameters += (len(self.values)**(i+1))
        return nr_parameters

