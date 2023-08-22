# Machine Learning Algorithms from Scratch

This project implements various machine learning algorithms from scratch using Python and NumPy, without relying on external libraries such as TensorFlow, Keras, or scikit-learn. The implemented algorithms include classification, regression, clustering, and basic neural network models.

## Table of Contents

- [Algorithms Implemented](#algorithms-implemented)
  - [Classification](#classification)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Logistic Regression](#logistic-regression)
    - [Naive Bayes](#naive-bayes)
  - [Regression](#regression)
    - [Linear Regression](#linear-regression)
  - [Clustering](#clustering)
    - [K-Means Clustering](#k-means-clustering)
  - [Neural Networks](#neural-networks)
    - [Perceptron](#perceptron)
    - [Neural Network](#neural-network)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributions and Feedback](#contributions-and-feedback)
- [Contact](#contact)

## Algorithms Implemented

### Classification
1. **K-Nearest Neighbors (KNN)**
   - The KNN algorithm classifies data points based on the majority class of their k-nearest neighbors in the feature space.

2. **Logistic Regression**
   - Logistic Regression is a binary classification algorithm that models the probability of a data point belonging to a particular class using a logistic function.

3. **Naive Bayes**
   - The Naive Bayes algorithm is a probabilistic classifier based on Bayes' theorem with the assumption of independence between features.

### Regression
4. **Linear Regression**
   - Linear Regression is used for predicting continuous numerical values by fitting a linear relationship between the input features and the target variable.

### Clustering
5. **K-Means Clustering**
   - K-Means is an unsupervised clustering algorithm that partitions data points into 'k' clusters based on their similarity in the feature space.

### Neural Networks
6. **Perceptron**
   - The Perceptron is a basic neural network unit that learns to classify inputs into two categories using a weighted sum and a threshold activation function.

7. **Neural Network**
   - This implementation represents a simple feedforward neural network with customizable architecture, including multiple layers and activation functions.

## Project Structure

- **/algorithms**
  - Contains implementation files for each algorithm (e.g., `KNN`, `KMeans`, ...).

- **Example.py**
  - An example usage script that demonstrates how to use each implemented algorithm on sample datasets.

- **README.md**
  - The main documentation file (this file) that provides an overview of the project, algorithms, and instructions for usage.


Otherwise, this is the repository tree:

- **/Classification**
  - **/KNN**
  - **/LogisticRegression**
  - **/NaiveBayes**

- **/Clustering**
  - **/KMeans**

- **/NeuralNetwork**
  - **/Perceptron**
  - **/NeuralNetwork**

- **/Regression**
  - **/LinearRegression**


## Getting Started

1. Clone this repository: `git clone https://github.com/your-username/Machine-Learning-From-Scratch.git`

2. Navigate to the project directory: `cd Machine-Learning-From-Scratch`

3. Install required packages: `pip install numpy matplotlib pandas`

4. Run the example script: `python Example.py`

## Usage

Each algorithm's implementation is contained within its respective file in the `/algorithms` directory. To use a specific algorithm, import the corresponding module in your code.


Example:
``` python 
from KNN import KNN

#Create KNN classifier
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

## Contributions and Feedback

If you find any issues, bugs, or ways to improve the implementations, please feel free to open an issue or pull request.

## Contact

By <a href="https://www.linkedin.com/in/aymane-maghouti/" target="_blank">Aymane Maghouti</a><br>






