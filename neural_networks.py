import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import k_means_helper as cluster
import linear_decision_boundaries as lin
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

def meanSquaredError(data, params, classes):
    sum = 0
    for n in range(len(data)):
        predicted = lin.neural_network(data[n], params[0], params[1])
        sum += (predicted - classToNumber(classes[n])) ** 2
    return sum/len(data)

def classToNumber(class_name):
    if class_name == 'versicolor':
        return 0
    return 1 #virginica

def gradient(data, params, classes):
    weights = params[0]
    new_weights_gradient = []
    sum = 0
    new_bias_gradient = 0
    for i in range(len(weights)):
        sum = 0
        for n in range(len(data)):
            sigmoid = lin.sigmoid(data[n], params[0], params[1])
            sum += (sigmoid - classToNumber(classes[n])) * sigmoid*(1-sigmoid)* data[n][i]
            new_bias_gradient += sigmoid - classToNumber(classes[n])
        new_weights_gradient.append(2/len(weights) * sum )
    return [new_weights_gradient, new_bias_gradient]

def make_step(data, params, classes, epsilon):
    weights = params[0]
    gradient_weight = gradient(data, params, classes)[0]
    gradient_bias = gradient(data, params, classes)[1]
    new_weights = []
    for i in range(len(weights)):
        new_weights.append(weights[i] -  epsilon * gradient_weight[i])
    new_bias = params[1] - gradient_bias* epsilon
    return [new_weights, new_bias] #make sure to calculate the new bias too

# returns the mean squared error
def neuralA(data, params, classes):
    x = meanSquaredError(data, params, classes)
    return x

def neural_B(inputs, weights1, weights2,  bias1, bias2 , classes):
    mean1 = meanSquaredError(inputs,[weights1, bias1], classes)
    title = f"Scatter Plot of Petal Length vs Petal Width"
    lin.boundaries_C(weights1, bias1, title + f"\n(Weights = {weights1} Mean Square Error = {mean1})")
    mean2 =(meanSquaredError(inputs,[weights2, bias2], classes ))
    lin.boundaries_C(weights2, bias2, title + f"\n(Weights = {weights2} Mean Square Error = {mean2})")

def neural_E(inputs, weights1, bias, classes, epsilon):
    title = ""
    lin.boundaries_C(weights1, bias, title + f"Step 1 (Weights = \n {weights1})")
    weights2 = make_step(inputs, [weights1, bias], classes, epsilon)
    lin.boundaries_C(weights2[0], weights2[1], title + f"Step 2 (Weights = \n {weights2[0]})")
    
    
if __name__ == "__main__":
    #zip together the values you want in your array (the number of input dimensions)
    inputs = list(zip(petal_length, petal_width))
    weights1 = np.array([0.5,0.5]) 
    weights2 = np.array([0.4,0.4]) 
    bias1 = 3.3
    bias2 = 3.3
    epsilon = 0.005
    
    # print(neuralA(inputs, [weights1, bias1], class_vals ))
    # neural_B(inputs, weights1, weights2,  bias1, bias2 , class_vals)
    # neural_E(inputs, weights2, bias1, class_vals, epsilon)
