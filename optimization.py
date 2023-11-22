#file to fun code for Exercise 4: Learning a Decision Boundary through Optimization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import k_means_helper as cluster
import linear_decision_boundaries as lin
import neural_networks as neural
# Read the CSV file into a DataFrame
df = pd.read_csv('irisdata.csv')

#filter out the first species
filtered_df = df.loc[df['species'].isin(['virginica', 'versicolor'])].copy()

# Extract columns for plotting
sepal_length = filtered_df['sepal_length']
sepal_width = filtered_df['sepal_width']
petal_length = filtered_df['petal_length']
petal_width = filtered_df['petal_width']
species = filtered_df['species']
class_vals = species.values

# Map species to numerical values
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
filtered_df['species_numeric'] = species.map(species_mapping)
def plot_learning_curve(prevErrors, data, params, classes):
    print(params)
    prevErrors.append(neural.meanSquaredError(data, params, classes))
    plt.plot(prevErrors,'go-', label='line 1', linewidth=2)
    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(f'Learning curve for iteration = {len(prevErrors)}')
    plt.ylim(0,0.5)
    plt.xlim(0,20)

    # Show the plot
    plt.show()
    return prevErrors
    
def optimize_A(inputs, weights1, bias, classes, epsilon, max_iterations):
    count = 1
    lin.boundaries_C(weights1, bias, f"Step {count} (Weights = \n {weights1})")
    weights2 = neural.make_step(inputs, [weights1, bias], classes, epsilon)
    count += 1
    lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
    weights1 = weights2
    count += 1
    weights2 = neural.make_step(inputs, [weights[0], weights1[1]], classes, epsilon)
    lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
    while (count < max_iterations) & (neural.meanSquaredError(inputs, [weights1[0], weights1[1]], classes) > 0.05 ):
        weights1 = weights2
        count += 1
        weights2 = neural.make_step(inputs, [weights1[0], weights1[1]], classes, epsilon)
        lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
    
def optimize_B(inputs, weights1, bias, classes, epsilon, max_iterations):
    prevErrors = []
    count = 1
    lin.boundaries_C(weights1, bias, f"Step {count} (Weights = \n {weights1})")
    prevErrors = plot_learning_curve(prevErrors, inputs, [weights1, bias], classes)
    weights2 = neural.make_step(inputs, [weights1, bias], classes, epsilon)
    count += 1
    lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
    prevErrors = plot_learning_curve(prevErrors, inputs, [weights2[0], weights2[1]], classes)
    weights1 = weights2
    count += 1
    weights2 = neural.make_step(inputs, [weights1[0], weights1[1]], classes, epsilon)
    lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
    prevErrors = plot_learning_curve(prevErrors, inputs, [weights2[0], weights2[1]], classes)
    while (count < max_iterations) & (neural.meanSquaredError(inputs, [weights1[0], weights1[1]], classes) > 0.04 ):
        weights1 = weights2
        count += 1
        weights2 = neural.make_step(inputs, [weights1[0], weights1[1]], classes, epsilon)
        lin.boundaries_C(weights2[0], weights2[1], f"Step {count} (Weights = \n {weights2[0]})")
        prevErrors = plot_learning_curve(prevErrors, inputs, [weights2[0], weights2[1]], classes)

    
if __name__ == "__main__":
    #zip together the values you want in your array (the number of input dimensions)
    inputs = list(zip(petal_length, petal_width))
    weights1 = np.array([0.5,0.5]) 
    weights2 = np.array([0.4,0.4]) 
    bias1 = 3.3
    bias2 = 3.3
    epsilon = 0.001
    max_iterations = 20
    
    # optimize_A(inputs, weights2, bias1, class_vals, epsilon, max_iterations)
    optimize_B(inputs, weights2, bias1, class_vals, epsilon, max_iterations)