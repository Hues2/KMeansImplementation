import copy
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# Get the data from the csv file
dataset = pd.read_csv('loan_data_set.csv')
data = dataset[['ApplicantIncome', 'LoanAmount']]

# This is a 2d array of the values from the 2 selected attributes
list_of_data_pre_processed = data.values
list_of_data = []
print(f"Size of data before pre-processing: {len(list_of_data_pre_processed)}")
for i in range(len(list_of_data_pre_processed)):
    if np.isnan(list_of_data_pre_processed[i][0]) or np.isnan(list_of_data_pre_processed[i][1]):
        pass
    else:
        list_of_data.append(list_of_data_pre_processed[i])

print(f"Size of data after pre-processing: {len(list_of_data)}")

# List of valid numbers for k
list_of_numbers = [1, 2, 3, 4, 5]
user_input = input("Enter the number of clusters (1-5), 3 is the default:  ")

if not user_input.isnumeric():
    print("Input is not a valid number. The default (3) will be used.")
    user_input = 3

k = int(user_input)
if k not in list_of_numbers:
    print("You entered an invalid number, the default number will be used")
    k = 3


# When this is true, the while loop will stop
centroids_are_the_same = False

# List of centroids
centroids = []

def printCentroids():
    for i in range(k):
        print(f'Centroid {i}: {centroids[i]}')

# List of colours (max 5)
list_of_colours_centroids = ['darkblue', 'black', 'purple', 'darkred', 'darkgreen']
list_of_colours_points = ['cornflowerblue', 'gray', 'violet', 'red', 'mediumseagreen']


# Get random centroids and add them to the list
for i in range(k):
    # Random number
    random_index = random.randint(0, (len(list_of_data) - 1))
    centroids.append(list_of_data[random_index])


# These are the randomly selected centroids
print("Initial Random Centroids:")
printCentroids()


# Plot the graph with the random centroids
plt.scatter(data['ApplicantIncome'], data['LoanAmount'], c='black')
for i in range(k):
    plt.scatter(centroids[i][0], centroids[i][1], c='red')
plt.title("Initial graph with random centroids")
plt.show()



# Calculate the euclidean distance between each point and the centroids

def euclidean_distance(point):
    # Loop through all the centroids
    # Measure the distance from each centroid from the point in each iteration
    # Add the distances to a list in order
    # Identify the smallest distance
    # Get the index of that smallest number
    # That index will indicate what centroid to return.
    # Return it with the f string
    # return (f"c{index}"

    distances = []
    for i in range(k):
        dist = np.sqrt((centroids[i][0] - point[0]) ** 2 + (centroids[i][1] - point[1]) ** 2)
        distances.append(dist)
    min_dist = min(distances)
    min_dist_index = distances.index(min_dist)
    return min_dist_index



def fill_clusters(list_of_clusters):
    for i in range(len(list_of_data)):
        result = euclidean_distance(list_of_data[i])
        list_of_clusters[result].append(list_of_data[i])



def get_new_centroids(list_of_clusters):
    for i in range(len(list_of_clusters)):
        total_x = 0
        total_y = 0
        average_x = 0
        average_y = 0
        for x in range(len(list_of_clusters[i])):
            total_x += list_of_clusters[i][x][0]
            total_y += list_of_clusters[i][x][1]
        average_x = total_x / len(list_of_clusters[i])
        average_y = total_y / len(list_of_clusters[i])
        averages.append(average_x)
        averages.append(average_y)
        new_centroids.append(copy.copy(averages))
        averages.clear()


def plot_final_graph(list_of_clusters, list_of_centroids):
    for i in range(k):
        for x in range(len(list_of_clusters[i])):
            plt.scatter(list_of_clusters[i][x][0], list_of_clusters[i][x][1], c=list_of_colours_points[i])
        plt.scatter(list_of_centroids[i][0], list_of_centroids[i][1], c=list_of_colours_centroids[i])

    plt.title("Final Graph")
    plt.show()



clusters = []

while not centroids_are_the_same:
    # Create list of lists of clusters, same amount as k
    clusters = [[] for i in range(k)]

    # Fill the clusters with the corresponding data points
    fill_clusters(clusters)

    # Get averages for x and y for each cluster and these will be the new centroids
    averages = []
    new_centroids = []
    get_new_centroids(clusters)

    # Convert new and old centroids to numpy arrays so that they can be checked to see if they are equal
    new_centroids_np = np.array(new_centroids)
    centroids_np = np.array(centroids)

    # Check to see if these new centroids are the same as the old ones
    for i in range(len(new_centroids_np)):
        if np.all(new_centroids_np == centroids_np):
            centroids_are_the_same = True
    centroids = new_centroids

    # Repeat these steps until the centroids don't change


# Final size of the clusters
print("Final size of each cluster:")
for i in range(len(clusters)):
    print(f"Size of cluster {i} : {len(clusters[i])}")

# Final Centroids:
print("Final Centroids")
printCentroids()


# Plot final clusters and centroids
plot_final_graph(clusters, centroids)




# Between Centroids in different clusters
# (I have a list of all centroids, so it's easy to get the distance between them all)
# Using the list of centroids --> centroids
# EXAMPLE --> THERE ARE 3 CLUSTERS,
# I CHECK THE DISTANCE BETWEEN THE FIRST CLUSTER WITH ALL OTHERS AFTER IT IN THE LIST
# THEN THE SECOND WITH THE ONE AFTER THAT
# WHEN IT GETS TO THE LAST ELEMENT IN THE LIST (THE LAST CENTROID), DON'T DO ANYTHING,
# AS ALL DISTANCES HAVE BEEN CHECKED

def min_separation():
    # More than one cluster
    if len(centroids) > 1:
        min_distance = np.sqrt((centroids[0][0] - centroids[1][0]) ** 2 + (centroids[0][1] - centroids[1][1]) ** 2)
        for i in range(k):
            j = i + 1
            while j != k:
                dist = np.sqrt((centroids[i][0] - centroids[j][0]) ** 2 + (centroids[i][1] - centroids[j][1]) ** 2)
                print(dist)
                if dist < min_distance:
                    min_distance = dist
                j += 1
    # Only one cluster, the distance will be 0
    else:
        min_distance = 0

    print(min_distance)
    return min_distance


# Between points in the same cluster
# Same as the min_separation
# Do the euclidean distance between all points in each cluster
# So loop k amount of times, and in the while loop do the same functionality
# (It will just do way more iterations )
# clusters is a list with each cluster in it
# and each cluster is a list with all the different data points

def max_compactness():
    max_distance = 0
    for i in range(k):
        j = 0
        while j != (len(clusters[i]) - 1):
            dist = np.sqrt((clusters[i][j][0] - clusters[i][j + 1][0]) ** 2 + (clusters[i][j][1] - clusters[i][j + 1][1]) ** 2)
            if dist > max_distance:
                max_distance = dist
            j += 1
    print(max_distance)
    return max_distance


def getDunnIndex():
    print("Dunn Index:")
    dunn_index = min_separation() / max_compactness()
    print(dunn_index)


getDunnIndex()


# DUN INDEX:

# Smallest euclidean distance between the centroids distances
# Biggest euclidean distance between 2 points in a cluster
