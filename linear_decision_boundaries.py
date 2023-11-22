import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import k_means_helper as cluster

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

# Map species to numerical values
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
filtered_df['species_numeric'] = species.map(species_mapping)


def boundaries_A(input):
    # Create a scatter plot
    plt.scatter(petal_length, petal_width, c=filtered_df['species_numeric'])

    # Add labels and title
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Scatter Plot of Petal Length vs Petal Width')

    # Show the plot
    plt.show()

# classifies the input as either class 0 or class 1 
# by returning a number between 0-1 
# later this number will either be rounded down to zero or up to 1 to classify the point
def sigmoid(x, weights, bias):
    sum = -1*bias
    for i in range(len(x)):
        sum += x[i]*weights[i]
    return 1 / (1 + np.exp(-sum))

#returns an array of length of the input data 
# classifies each input as either class 0 or class 1 (a range between 0-1 that will later be classified)
def neural_network(input, weights, bias):
    sig = (sigmoid(input, weights, bias))
    if sig < 0.5:
        return 0
    else:
        return 1
    

# Write a function that plots the decision boundary for the non-linearity above overlaid on the iris data.
# Choose a boundary (by setting weight parameters by hand) that roughly separates the two classes. Use
# an output of 0 to indicate the 2nd iris class, and 1 to indicate the 3rd.
def boundaries_C(weights, bias, title = "Scatter Plot of Petal Length vs Petal Width"):
    # using countourf, we can plot the decision boundary where p(C1) = p(C2) 
    # assuming the threshold is 0.5
    x_values = petal_length
    y_values = petal_width
    # Create a mesh grid
    x_min, x_max = x_values.min() - 1, x_values.max() + 1
    y_min, y_max = y_values.min() - 1, y_values.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    
    # Apply sigmoid function to each point in the mesh grid
    Z = np.array([sigmoid([x, y], weights, bias) for x, y in zip(np.ravel(xx), np.ravel(yy))])

    # Reshape Z for contour plot
    Z = Z.reshape(xx.shape)
    
    #plot contour
    plt.contour(xx, yy, Z, levels=[0, 0.5, 1], colors='blue', linewidths=2)
    # Create a scatter plot
    plt.scatter(petal_length, petal_width, c=filtered_df['species_numeric'])

    # Add labels and title
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title(title)

    # Show the plot
    plt.show()

def boundaries_D(weights, bias):
    # using countourf, we can plot the decision boundary where p(C1) = p(C2) 
    # assuming the threshold is 0.5
    x_values = petal_length
    y_values = petal_width
    # Create a mesh grid
    x = np.linspace(-10, 10, 100) 
    y = np.linspace(-10, 10, 100) 
    xx, yy = np.meshgrid(x, y)
    
    # Apply sigmoid function to each point in the mesh grid
    Z = np.array([sigmoid([x, y], weights, bias) for x, y in zip(np.ravel(xx), np.ravel(yy))])

    # Reshape Z for contour plot
    Z = Z.reshape(xx.shape)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(xx, yy, Z, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_zlabel('Logistic function Value')
    ax.set_title('Scatter Plot of Petal Length vs Petal Width')

    # Show the plot
    plt.show()

def boundaries_E(testpoints, weights, bias):
    
    output = []
    petal_l = []
    petal_w = []
    for i in range(len(testpoints)):
        output.append(neural_network(testpoints[i], weights, bias))
        petal_l.append(testpoints[i][0])
        petal_w.append(testpoints[i][1])
    
    df = pd.DataFrame({
        'Petal Length (cm)': petal_l,
        'Petal Width (cm)': petal_w,
        'Output(0 = veriscolor, 1 = virginica)': output
    })
    # Plot the table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')  # Turn off axis

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Plotting output for sample inputs')
    table.auto_set_column_width([0, 1, 2])
    plt.show()
    
if __name__ == "__main__":
    #zip together the values you want in your array (the number of input dimensions)
    inputs = list(zip(petal_length, petal_width))
    weights_matrix = np.array([0.5,0.5]) 
    bias = 3.3
    
    testpoints = [[petal_length[135], petal_width[135]], #ambigious
                [petal_length[59], petal_width[59]], #veriscolor
                [petal_length[115], petal_width[115]], #ambigious
                [petal_length[120], petal_width[120]] #virginica
                  ]
    # boundaries_A(inputs)
    # boundaries_C(weights_matrix, bias)
    # boundaries_D(weights_matrix, bias)
    boundaries_E(testpoints, weights_matrix, bias)