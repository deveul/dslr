import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cost_history(costs):
    plt.figure()
    sns.set_style('white')
    colors = ['r', 'g', 'b', 'gold']
    for i, cost in enumerate(costs):
        plt.plot(range(len(cost)), cost, c=colors[i])
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

def scatter_data(self, X, y):
    sns.set_style('white')
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y.reshape(-1))
    plt.show()

def plot_result(self, params_optimal, X, y):
    slope = -(params_optimal[1] / params_optimal[2])
    intercept = -(params_optimal[0] / params_optimal[2])
    
    sns.set_style('white')
    sns.scatterplot(x=X[:,1],y=X[:,2],hue=y.reshape(-1))
    
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (slope * x_vals)
    plt.plot(x_vals, y_vals, c="k")
    plt.show()
