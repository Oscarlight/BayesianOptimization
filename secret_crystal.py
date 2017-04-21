'''
    Create secret crystal
    Credit: 
'''

import sys
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 16
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
mpl.rcParams['figure.figsize'] = 10, 6
# for predictable result
np.random.seed(42)
# temperature
xs = np.linspace(-2, 10, 10000)
# secret function
f = 8 * (np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1))

plt.figure()
plt.xlim(0, len(f))
plt.ylim(f.min()-0.1*(f.max()-f.min()), f.max()+0.1*(f.max()-f.min()))
plt.plot(f)
plt.xlabel('Growth Temperature (K)')
plt.ylabel('Hardness (10 is the hardness of Vibranium)')
plt.savefig('secret_curve.png')

def plot_bo(f, bo):
    xs = [x["x"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]

    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)
    
    plt.figure()
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean+sigma, mean-sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min()-0.1*(f.max()-f.min()), f.max()+0.1*(f.max()-f.min()))
    plt.xlabel('Growth Temperature (K)')
    plt.ylabel('Hardness (10 is the hardness of Vibranium)')
    plt.savefig('gpr_result.png')

# use sklearn's default parameters for theta and random_start
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
bo = BayesianOptimization(f=lambda x: f[int(x)],
                          pbounds={"x": (0, len(f)-1)},
                          verbose=0)

bo.maximize(init_points=2, n_iter=4, acq="ucb", kappa=5, **gp_params)
plot_bo(f, bo)
