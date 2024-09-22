import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# Load the dataset
iris = pd.read_csv("iris.csv")

# Display the first few rows of the dataset
print(iris.head())

# Display descriptive statistics
print(iris.describe())

# Check unique species
print("Target Labels:", iris["species"].unique())

# Visualize the data
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Prepare the data for training
X = iris.drop("species", axis=1)
y = iris["species"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict the species for a new sample
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction:", prediction)