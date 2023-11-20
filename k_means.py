import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import k_means_helper as cluster

# Read the CSV file into a DataFrame
df = pd.read_csv('irisdata.csv')

# Extract columns for plotting
sepal_length = df['sepal_length']
sepal_width = df['sepal_width']
petal_length = df['petal_length']
petal_width = df['petal_width']
species = df['species']

# Define colors for each species
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

def clustering_A(input, k):
    #k_means!
    clustered_array = cluster.seed_k(input, k)

    # Create a scatter plot
    plt.scatter(petal_length, petal_width, c=clustered_array)

    # Add labels and title
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Scatter Plot of Petal Length vs Petal Width')

    # Show the plot
    plt.show()
    
def clustering_B(input, k):
    #k_means!
    clustered_array = np.array(cluster.seed_k_objective_function(input, k))
    # Create a scatter plot
    plt.plot(clustered_array[:,0],clustered_array[:,1])

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value (D)')
    plt.title('Objective Function Value vs. Iterations')

    # Show the plot
    plt.show()

def clustering_C(k):
    cluster.seed_k_plot_intermediates(list(zip(petal_length, petal_width)), k, petal_length, petal_width);

def clustering_D(k):
    input = list(zip(petal_length, petal_width))
    #k_means!
    clustered_array = cluster.seed_k_decision_boundaries(input, k)
    
    # Create a scatter plot
    plt.scatter(petal_length, petal_width, c=clustered_array[0])
    plt.scatter(np.array(clustered_array[2])[:,0], np.array(clustered_array[2])[:,1], label='Weighted Means', 
                    marker='x', color='red', s=50)
    if k==3:
        plt.plot(clustered_array[1][0][0], clustered_array[1][0][1], label='Decision Boundary 1')
        plt.plot(clustered_array[1][1][0], clustered_array[1][1][1], label='Decision Boundary 2')
    else: # k == 2
        plt.plot(clustered_array[1][0][0], clustered_array[1][0][1], label='Decision Boundary 1')
    # Add labels and title
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title(f'Scatter Plot of Petal Length vs Petal Width (k = {k})')
    
    # Set y-axis limits
    plt.ylim(0, 2.6)
    # Show the plot
    plt.show()

if __name__ == "__main__":
    #zip together the values you want in your array (the number of input dimensions)
    inputs = list(zip(sepal_length, sepal_width, petal_length, petal_width))
    k = 2
    #clustering_A(inputs, k)
    # clustering_B(inputs, k)
    # clustering_C(k)
    clustering_D(k)