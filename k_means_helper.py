import random
import colorsys
import matplotlib.pyplot as plt
import numpy as np

#picks k random points to use as our starter weighted_means
def select_k_random_points(points, k):
    random.seed(40)
    random_selection = random.sample(points, k)
    return random_selection

#generates k random colors to use for our legend
def generate_random_colors(k):
    random_colors = []

    for _ in range(k):
        # Generate random RGB values
        r, g, b = [random.uniform(0, 1) for _ in range(3)]

        # Convert RGB to hexadecimal
        hex_color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

        random_colors.append(hex_color)

    return random_colors

# assigns clusters by determining which weighted_mean is closest to each point
def assign_clusters(points, weighted_means):
    # points is an n by i matrix, weighted means is a k by i matrix
    classification_map = []
    for n in range(len(points)): 
        #d is how we compare the points to the weighted mean
        d = float('inf') 
        #curr_weighted_mean is the index of the weighted means that is closest to the x_n point
        curr_weighted_mean = 0
        for k in range(len(weighted_means)): 
            temp_d = 0 
            for i in range(len(points[0])):
                temp_d += (points[n][i] - weighted_means[k][i]) ** 2
            if temp_d < d:
                #found better classification for point x_n
                d = temp_d
                curr_weighted_mean = k
        classification_map.append(curr_weighted_mean)
    return classification_map
                
def update_means(classification_map, points, prev_weighted_means):
    new_weighted_means = []
    for k in range(len(prev_weighted_means)):
        curr_weighted_mean = []
        for i in range(len(prev_weighted_means[0])):
            top_sum = 0
            bottom_sum = 0
            for n in range(len(points)):
                #check if x_n is in cluster k (represents r_nk)
                if classification_map[n] == k:
                    top_sum += points[n][i]
                    bottom_sum += 1
            curr_weighted_mean.append(top_sum / bottom_sum)
        new_weighted_means.append(curr_weighted_mean)
    return new_weighted_means

def objective_fn_value(classification_map, points, weighted_means):
    sum = 0
    for n in range(len(classification_map)):
        for k in range(len(weighted_means)):
            if classification_map[n] == k:
                #find magnitude of x_n - mu_k (squared)
                for i in range(len(weighted_means[0])):
                    sum += abs(points[n][i] - weighted_means[k][i])
    return sum
                    
#main function to run kmeans (Q1 part A)
#takes as an input "points", which is the dataset with an arbitrary number of input dimensions
#takes as an input "k", which is the number of clusters you want the data to be classified as
def seed_k(points, k):
    prev_weighted_means = select_k_random_points(points,k)
    prev_classification_map = assign_clusters(points, prev_weighted_means)
    new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
    new_classification_map = assign_clusters(points, new_weighted_means)
    while(prev_classification_map != new_classification_map):
        prev_classification_map = new_classification_map
        prev_weighted_means = new_weighted_means
        new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
        new_classification_map = assign_clusters(points, new_weighted_means)
    return new_classification_map

#main function to run Q1 part B
def seed_k_objective_function(points, k):
    arr = []
    iterations = 0
    prev_weighted_means = select_k_random_points(points,k)
    prev_classification_map = assign_clusters(points, prev_weighted_means)
    arr.append([iterations, objective_fn_value(prev_classification_map, points, prev_weighted_means)])
    iterations += 1
    
    new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
    new_classification_map = assign_clusters(points, new_weighted_means)
    arr.append([iterations, objective_fn_value(new_classification_map, points, new_weighted_means)])
    iterations += 1
    
    while(prev_classification_map != new_classification_map):
        prev_classification_map = new_classification_map
        prev_weighted_means = new_weighted_means
        new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
        new_classification_map = assign_clusters(points, new_weighted_means)
        arr.append([iterations, objective_fn_value(new_classification_map, points, new_weighted_means)])
        iterations += 1
        
    return arr

def decision_boundary_between(point1,point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    slope = (y2-y1) / (x2-x1)
    perpendicular_slope = -1 / slope
    midpoint_x = (x1 + x2 )/2
    midpoint_y = (y1 + y2)/2
    x_vals = np.linspace(min(point1[0], point2[0]) - 1, max(point1[0], point2[0]) + 1, 100)
    y_vals = perpendicular_slope * (x_vals - midpoint_x) + midpoint_y
    # Plot the perpendicular line
    #plt.plot(x_vals, y_vals, label='Perpendicular Line')
    return [x_vals, y_vals]

#returns an array of points to find the decision boundary between
def find_decision_boundary_points(clusters):
    # Three points
    point0 = np.array([clusters[0][0], clusters[0][1]])
    point1 = np.array([clusters[1][0], clusters[1][1]])
    point2 = np.array([clusters[2][0], clusters[2][1]])

    # Check which point is between the other two
    if (
        (point0[0] <= point1[0] <= point2[0] or point2[0] <= point1[0] <= point0[0])
        and (point0[1] <= point1[1] <= point2[1] or point2[1] <= point1[1] <= point0[1])
    ):
        array_ordering = [[point0,point1],[point1,point2]]
        
    elif (
        (point1[0] <= point0[0] <= point2[0] or point2[0] <= point0[0] <= point1[0])
        and (point1[1] <= point0[1] <= point2[1] or point2[1] <= point0[1] <= point1[1])
    ):
        array_ordering = [[point0,point1],[point0,point2]]
    else:
        array_ordering = [[point0,point2],[point1,point2]]

    return array_ordering

#main function to run kmeans (Q1 Part C)
def seed_k_plot_intermediates(points, k, petal_length, petal_width):
    # Create a figure with multiple subplots using plt.subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    prev_weighted_means = select_k_random_points(points,k)
    prev_classification_map = assign_clusters(points, prev_weighted_means)
    axes[0].scatter(petal_length, petal_width, c=prev_classification_map, label='Dataset')
    axes[0].scatter(np.array(prev_weighted_means)[:,0], np.array(prev_weighted_means)[:,1], label='Weighted Means', 
                    marker='x', color='red', s=50)
    axes[0].set_title(f'Initial (k={k})')
    axes[0].legend()
    
    new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
    new_classification_map = assign_clusters(points, new_weighted_means)
    axes[1].scatter(petal_length, petal_width, c=new_classification_map, label='Dataset')
    axes[1].scatter(np.array(new_weighted_means)[:,0], np.array(new_weighted_means)[:,1], label='Weighted Means', 
                    marker='x', color='red', s=50)
    axes[1].set_title(f'Intermediate (Step 2, k={k})')
    axes[1].legend()
    while(prev_classification_map != new_classification_map):
        prev_classification_map = new_classification_map
        prev_weighted_means = new_weighted_means
        new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
        new_classification_map = assign_clusters(points, new_weighted_means)
        
    axes[2].scatter(petal_length, petal_width, c=new_classification_map, label='Dataset')
    axes[2].scatter(np.array(new_weighted_means)[:,0], np.array(new_weighted_means)[:,1], label='Weighted Means', 
                    marker='x', color='red', s=50)
    axes[2].set_title(f'Converged Clusters (k={k})')
    axes[2].legend()
    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

#main function to run kmeans (Q1 part D)
def seed_k_decision_boundaries(points, k):
    prev_weighted_means = select_k_random_points(points,k)
    prev_classification_map = assign_clusters(points, prev_weighted_means)
    new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
    new_classification_map = assign_clusters(points, new_weighted_means)
    while(prev_classification_map != new_classification_map):
        prev_classification_map = new_classification_map
        prev_weighted_means = new_weighted_means
        new_weighted_means = update_means(prev_classification_map, points, prev_weighted_means)
        new_classification_map = assign_clusters(points, new_weighted_means)
    
    boundary_arr = [];
    if k == 3:
        cluster_points = find_decision_boundary_points(new_weighted_means)
        boundary1 = decision_boundary_between(cluster_points[0][0],cluster_points [0][1] )
        boundary2 = decision_boundary_between(cluster_points[1][0],cluster_points [1][1] )
        boundary_arr.append(boundary1)
        boundary_arr.append(boundary2)
    else: #k == 2
        boundary1 = decision_boundary_between(new_weighted_means[0], new_weighted_means[1])
        boundary_arr.append(boundary1)
        
    return [new_classification_map, boundary_arr, new_weighted_means]