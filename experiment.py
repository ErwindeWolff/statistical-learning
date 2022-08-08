from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import os


def read_data(filename, delimiter=";"):
    '''
        This function reads in a CSV file delimitered by
        the delimiter character (default = semi-colon).
        The file should have the following format:
        triplet_name;shape;rt

        RETURNS triplet_names (string list)
                shapes (string list)
                rts (float list)
    '''
    triplet_names = []
    shapes = []
    rts = []

    # Read in all the triplet types, shapes and response times
    f = open(filename)
    for i, line in enumerate(f):
        if i > 0:
            terms = [x.strip() for x in line.split(delimiter) if len(x) > 0]
            triplet_names.append(terms[0])
            shapes.append(terms[1])
            rts.append(float(terms[2]))
    f.close()

    # Remove RTs shorter than 100 ms or bigger than mean + 3*std
    p_mean = np.mean(rts, axis=0)
    p_std = np.std(rts, axis=0)
    for i in range(len(rts)):
        rt = rts[i]
        if rt < 100 or (rt > p_mean+3*p_std):
            rts[i] = np.nan

    # Create log-transformed versions of true RTs, then Z-score
    rts = np.log(np.array(rts))
    rts = (rts - np.nanmean(rts))/np.nanstd(rts)
            
    return triplet_names, shapes, rts



def run_experiment(agent, shapes):
    '''
        Optional argument: convert_rts. If True, this will make sure the
        predicted RTs are in same range as the observed RTs.

        Shows the shapes to the agent one by one.
        Before each shape, the prior prediction is asked.
        After the shape, the response time is calculated
        as the PE of seeing that shape given the prior belief,
        plus the entropy of the posterior belief.

        Returns a vector of predicted response times.
    '''
    predicted_rts = []
    log_likelihoods = []

    for shape in shapes:
        # Get the prior belief
        prior = agent.get_probabilities()

        # Convert shape name into a one-hot list
        obs = [0 for _ in prior]
        index = agent.values.index(shape)
        obs[index] = 1

        # Update the log-likelihood (for BIC)
        log_likelihoods.append(np.log(prior[index]))

        # Let agent learns from this observation
        agent.process_observation(shape)
        
        # Get the posterior belief
        posterior = agent.get_probabilities()

        # Determine predicted RT
        pred_rt = entropy(obs, qk=prior, base=2)
        
        predicted_rts.append(pred_rt)

    return predicted_rts, log_likelihoods



def create_posterior_images(agents, filename, posterior, triplet_types, triplet_posterior,
                            posterior_se=None, triplet_posterior_se=None):
    '''
        This function makes the graphs of all the relevant data
        given the response times and predicted response times as a vector,
        and the posterior probabilities and triplet posteriors as a (AxN) matrix,
        where A is the number of agents and N is the number of triplets seen.
        The graphs are saved in the results folder. This is created if non-existent
        when calling this function.
    '''

    # Make sure that there is a results folder
    if not(os.path.exists("results")):
        os.mkdir("results")

    # Create the legend for the image being the different agents
    legend = [agent.name.replace("_", " ").title() for agent in agents]

    # Create name for files
    name = filename.replace(".csv", "")
    x_label = "Number of Seen Shapes"

    # Save Posterior over time
    if posterior.shape[-1] != 0:
        plt.figure()
        plt.title("Posterior Probability Over Time")
        plt.plot(posterior.T)
        if not(posterior_se is None):
            x = [n+1 for n in range(posterior.shape[-1])]
            for i in range(posterior.shape[0]):
                plt.fill_between(x, posterior[i] - posterior_se[i],
                                 posterior[i] + posterior_se[i], alpha=0.2)
            
        plt.xlabel(x_label)
        plt.ylabel("Probability")
        plt.legend(legend)
        plt.ylim([-0.1, 1.1])
        plt.savefig(f"results/{name}_posterior.png")
        plt.close()

    # Save posterior probability per triplet type
    if triplet_posterior.shape[-1] != 0:
        for i, target in enumerate(triplet_types):
            text = target.replace("_", " ").replace("type","input structure").title()
            plt.figure()
            plt.title(f"Posterior Probability for {text} Triplets Over Time")
            plt.plot(triplet_posterior[i].T)
            if not(triplet_posterior_se is None):
                x = [n+1 for n in range(triplet_posterior.shape[-1])]
                for j in range(triplet_posterior.shape[1]):
                    plt.fill_between(x, triplet_posterior[i,j] - triplet_posterior_se[i,j],
                                     triplet_posterior[i,j] + triplet_posterior_se[i,j], alpha=0.2)

            plt.xlabel("Number of Seen Shapes")
            plt.ylabel("Probability")
            plt.legend(legend)
            plt.ylim([-0.1, 1.1])
            plt.savefig(f"results/{name}_{target}_triplets_posterior.png")
            plt.close()


def create_rt_images(agents, filename, true_rts, pred_rts):
    # Make sure that there is a results folder
    if not(os.path.exists("results")):
        os.mkdir("results")

    # Create the legend for the image being the different agents
    legend = [agent.name.replace("_", " ").title() for agent in agents]
    legend += ["Participant"]

    # Create name for files
    name = filename.replace(".csv", "")
    x_label = "Number of Seen Shapes"

    # Save log response times
    plt.figure()
    plt.title("RT Predictions Over Time")
    plt.plot(pred_rts.T)
    plt.plot(true_rts)
    plt.xlabel(x_label)
    plt.ylabel("Normalized Response Time")
    plt.legend(legend)
    plt.savefig(f"results/{name}_model_curves.png")
    plt.close()


def create_files(agents, filename, full_posteriors, triplet_types, full_triplet_posteriors):
    '''
        This function makes the text files of all the relevant data
        given the posterior probabilities and triplet posteriors as a (AxN) matrix,
        where A is the number of agents and N is the number of triplets seen.
        The text files are saved in the results folder. This is created if
        non-existent when calling this function.
    '''
    
    # Make sure that there is a results folder
    if not(os.path.exists("results")):
        os.mkdir("results")

    name = filename.replace(".csv", "")
    
    # Create text file of global posterior
    posteriors = full_posteriors.T
    f = open("results/" + name + "_posteriors.txt", "w")
    f.write(";".join([agent.name for agent in agents]) + "\n")
    for data in posteriors:
        f.write(";".join([str(d) for d in data]) + "\n")
    f.close()

    # Create text file of triplet posterior
    for target, triplet_posteriors in zip(triplet_types, full_triplet_posteriors):
        posteriors = triplet_posteriors.T
        f = open("results/" + name + "_" + target + "_posteriors.txt", "w")
        f.write(";".join([agent.name for agent in agents]) + "\n")
        for data in posteriors:
            f.write(";".join([str(d) for d in data]) + "\n")
        f.close()




