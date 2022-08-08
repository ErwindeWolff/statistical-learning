class BaselineLearner():

    def __init__(self, values):
        self.name = "baseline"
        self.values = values

        # Create uniform prediction
        self.prediction = [1/len(values) for _ in values]

    def reset(self):
        '''
            Resets all learning so far.
        '''
        pass


    def process_observation(self, obs):
        pass


    def get_probabilities(self):
        '''
            Get the next prediction. This is automatically updated given
            a series of shapes. I.e. no argument is needed for this method.
        '''
        return self.prediction


    def get_number_parameters(self):
        '''
            Returns the number of parameters used by the agent.
            This is used to calculate the BIC score
        '''
        return 0
