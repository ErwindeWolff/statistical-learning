import experiment
import numpy as np
import matplotlib.pyplot as plt
import os

from Model_Baseline import *
from Model_Chunking import *
from Model_Connected import *
from Model_Disconnected import *
from Model_Conjunctive import *
from Model_TP import *

from math import log, sqrt
from scipy.stats import norm, zscore


#######################################################################################
#######################################################################################
#######################################################################################
'''
    This file can simply be ran.
    It will then go through each .txt file in the data folder.
    This data folder should be placed in the same folder as this script.
    For each file, it will create a separate series of result images and text files.
    In addition, it will average the posteriors per participant, and put those result
    in an image and text file as well.
'''
#######################################################################################
#######################################################################################
#######################################################################################

def compare_rts(all_pred_rts, true_rts):
    
    # Gain likelihoods for each agent
    likelihoods = norm.pdf(all_pred_rts, loc=true_rts)
    likelihoods[np.where(np.isnan(likelihoods))] = 1

    # Determine posteriors over time
    posteriors = likelihoods.copy()
    posteriors[:,0] = posteriors[:,0]/np.sum(posteriors[:,0])
    for i in range(likelihoods.shape[1]-1):
        posteriors[:,i+1] = posteriors[:,i] * posteriors[:,i+1]
        posteriors[:,i+1] = posteriors[:,i+1]/np.sum(posteriors[:,i+1])
        
    return posteriors
    

def process_data(agents, triplet_names, shapes, true_rts, filename):
    # Make sure all the agents start with a blank slate
    for agent in agents:
        agent.reset()
    
    # Get the predicted RTs for each agent given the data
    all_pred_rts = np.zeros((len(agents), len(true_rts)))
    for i, agent in enumerate(agents):
        pred_rts, _ = experiment.run_experiment(agent, shapes)
        all_pred_rts[i, :] = np.array(pred_rts)

    # Perform zero-mean, unit-variance scaling (Z-scoring)
    all_pred_rts = np.nan_to_num(zscore(all_pred_rts, axis=1))

    # Get the posterior distributions overall
    posteriors = compare_rts(all_pred_rts, true_rts)

    # Determine posterior development per triplet type
    triplet_types = sorted(list(set(triplet_names)))
    triplet_posteriors = []
    triplet_pred_rts = []
    triplet_true_rts = []
    for triplet_type in triplet_types:
        indices = np.squeeze(np.where(np.array(triplet_names) == triplet_type))
        triplet_posterior = compare_rts(all_pred_rts[:, indices], true_rts[indices])
        triplet_posteriors.append(triplet_posterior)

        triplet_pred_rts.append(all_pred_rts[:,indices])
        triplet_true_rts.append(true_rts[indices])
        
    triplet_posteriors = np.array(triplet_posteriors)
    #triplet_posteriors = np.squeeze(triplet_posteriors)
        
    # Save the images
    experiment.create_posterior_images(agents, filename, posteriors, triplet_types, triplet_posteriors)
    #experiment.create_rt_images(agents, filename, true_rts, all_pred_rts)

    # Save the data in a file
    experiment.create_files(agents, filename, posteriors, triplet_types, triplet_posteriors)

    return all_pred_rts, true_rts, posteriors, triplet_posteriors #np.array(triplet_pred_rts), np.array(triplet_true_rts)




folder = "data/"

nr_files = 0
nr_agents = 0
nr_stimuli = 0
nr_triplet_types = 0

full_posterior = []
full_triplet_posterior = []
full_pred_rts = []
full_true_rts = []

print("Processing files...")
for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        nr_files += 1

n = 0
for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        print(f"\tFile {filename} ({n+1}/{nr_files})")
        
        triplet_names, shapes, true_rts = experiment.read_data(folder + filename, delimiter=",")

        values = list(set(shapes))
        agents = [TPLearner(values),
                  JointChunkLearner(values),
                  ConnectedChunkLearner(values),
                  DisconnectedChunkLearner(values),
                  ConjunctiveChunkLearner(values),
                  BaselineLearner(values),
                  ]

        # Process the data through the agents
        pred_rts, true_rts, posterior, triplet_posterior = process_data(agents, triplet_names, shapes, true_rts, filename)

        # Create empty matrices after first experiment
        if n == 0:
            nr_agents = len(agents)
            nr_stimuli = pred_rts.shape[1]
            triplet_types = sorted(list(set(triplet_names)))
            nr_triplet_types = len(triplet_types)
            
            full_triplet_types = triplet_types
            full_pred_rts = np.zeros((nr_files, nr_agents, nr_stimuli))
            full_true_rts = np.zeros((nr_files, nr_stimuli))

            full_posterior = np.zeros((nr_files, nr_agents, nr_stimuli))
            full_triplet_posterior = np.zeros((nr_files, nr_triplet_types, nr_agents,
                                               int(nr_stimuli/nr_triplet_types)))

        # Save the data from this file
        full_pred_rts[n,:,:] = pred_rts
        full_true_rts[n,:] = true_rts

        full_posterior[n,:,:] = posterior
        full_triplet_posterior[n,:,:,:] = triplet_posterior

        n += 1


print("Creating posterior files across participants...")
# Average the posterior probabilities
# and extract the SE
avg_posterior = np.mean(full_posterior, axis=0)
posterior_se = np.std(full_posterior, axis=0) / np.sqrt(n)

# Average the triplet posterior probabilities
avg_triplet_posterior = np.mean(full_triplet_posterior, axis=0)
triplet_posterior_se = np.std(full_triplet_posterior, axis=0) / np.sqrt(n)

# Average the predicted response times
avg_pred_rts = np.mean(full_pred_rts, axis=0)

# Average the true response times
avg_true_rts = np.nanmean(full_true_rts, axis=0)


experiment.create_posterior_images(agents, "general", avg_posterior,
                                   triplet_types, avg_triplet_posterior,
                                   posterior_se, triplet_posterior_se)

experiment.create_rt_images(agents, "general", avg_true_rts, avg_pred_rts)
experiment.create_files(agents, "general", avg_posterior, triplet_types,
                        avg_triplet_posterior)
print("Done!")





