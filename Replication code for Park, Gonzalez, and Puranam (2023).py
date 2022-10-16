# Import packages
import pandas as pd
import seaborn as sns
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Read data file
# Data structure: col1 = ID, col2 = round, col3 = choice, col4 = outcome
data = pd.read_csv(r'C:\Desktop\individual.csv')
#data = pd.read_csv(r'C:\Desktop\Groups with DDM.csv')
#data = pd.read_csv(r'C:\Desktop\Groups with CDM.csv')

# Parameter setting
ndim_phi = 100 # The number of grids for phi
ndim_tau = 100 # The number of grids for tau
samplesize = 1000 # The sample size for random sampling for priors; assign 1 for fixed priors
T = 100        # The number of rounds (fixed)
nsample = 56   # The number of subjects; (Individual = 54, Group with DDM = 56, Group with CDM = 54)

# Dataframe for outcome
result = pd.DataFrame(columns = ['phi', 'tau', 'Log-likelihood'])
phi_grid = []
tau_grid = []
Likelihood = []
heatmap = []

# Gridsearch
for i in tqdm(range(ndim_phi)):
    phi = 0.01 + i/ndim_phi
    for j in range(ndim_tau):
        tau = 0.05 + j/ndim_tau # Increase the minimum value when you encounter inf
        summ = 0
        for k in range(nsample):
            for sample in range(samplesize):
                for t in range(T):
                    if t == 0: # Initialize beliefs
                        #belief = [0.1+random.uniform(0,0.8), 0.1+random.uniform(0,0.8)] # Randomized priors; need greater samplesize and take longer 
                        belief = [0.5, 0.5]                                              # priors are fixed at 0.5; samplesize=1
                    expbelief = [math.exp(belief[0]/tau),math.exp(belief[1]/tau)] 
                    prob_A = expbelief[0]/(expbelief[0] + expbelief[1]) # Soft-max choice rule
                    if data.iloc[k*100+t,2] == 0:
                        summ += np.log(prob_A) # Add likelihood
                        belief[0] = (1-phi)*belief[0] + phi*data.iloc[k*100+t,3] # Exponential weighted averaging
                    else:
                        summ += np.log(1-prob_A) # Add likelihood
                        belief[1] = (1-phi)*belief[1] + phi*data.iloc[k*100+t,3] # Exponential weighted averaging
        phi_grid.append("{:.2f}".format(phi)) # Store parameter
        tau_grid.append("{:.2f}".format(tau)) # Store parameter
        Likelihood.append(summ) # Store likelihood
result['phi'] = phi_grid
result['tau'] = tau_grid
result['Log-likelihood'] = Likelihood

# Calculate Log-likelihood ratio
maxLL = max(result['Log-likelihood'])       # Get maximum likelihood
z = (-2)*(result['Log-likelihood'] - maxLL) # Get log-likelihood ratio
z[z>50]= np.nan                             # Replace LLR greater than 50 with nan for heatmap
result['Log-likelihood Ratio'] = z          # Store log-likelihood ratio

# Optimal parameters
optimal = result[result['Log-likelihood Ratio'].values == 0]
print(optimal)

# Plot heatmap
heatmap_data = pd.pivot_table(result[['phi','tau','Log-likelihood Ratio']], values='Log-likelihood Ratio', index=['tau'], columns='phi')
sns.heatmap(heatmap_data,cmap="coolwarm")
plt.xlabel("phi", size=14)
plt.ylabel("tau", size=14)
plt.title("Grid Search", size=14)
plt.tight_layout()


