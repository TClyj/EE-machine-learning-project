# Analyzing and Modeling Data Using a Three-Layer Feed-Forward Neural Network and Comparing Model Performance

Author: Yijia Lu

### Abstract: 


## Introduction and Overview



## Theoretical Background

**SVD** is a factorization of a matrix that can be used to analyze the underlying structure and patterns in the data, as well as for dimensionality reduction and feature extraction purposes. SVD decomposes a matrix X into three matrices: U, s (a diagonal matrix with singular values, usually represented as a 1D array), and V^T.

**LDA** is a linear classification algorithm that projects data points onto a lower-dimensional space to maximize the separation between different classes. It assumes that the data is normally distributed and that the covariances for each class are identical. LDA works by finding the linear combination of features that best separates the classes.

In the questions, LDA was used to classify pairs and triplets of digits from the MNIST dataset. By training the LDA classifier on the filtered training data (consisting of only the selected digits), we were able to find the linear decision boundary that best separates the chosen digits. The accuracy of the LDA classifier was then evaluated on the test data.

**SVM** is a powerful classification algorithm that finds the optimal hyperplane separating the data into different classes. The algorithm aims to maximize the margin between the hyperplane and the closest data points from each class, known as support vectors. SVM can handle both linear and non-linear classification tasks using kernel functions.

In the questions, SVM was applied to classify all ten digits in the MNIST dataset. The SVM classifier was trained on the entire training data, and its accuracy was evaluated on the test data. A comparison between LDA, SVM, and Decision Trees was then performed on the hardest and easiest pairs of digits to separate.

**Decision Trees** are a class of algorithms that recursively split the data into subsets based on feature values. They create a tree-like structure, where each internal node represents a decision based on a feature, and each leaf node represents the predicted class label. Decision Trees are simple to understand, can handle both categorical and continuous features, and are resistant to noise.

In the questions, the Decision Tree classifier was applied to classify all ten digits in the MNIST dataset. Similar to the SVM classifier, the Decision Tree was trained on the entire training data, and its accuracy was evaluated on the test data. The performance of the Decision Tree classifier was then compared to LDA and SVM on the hardest and easiest pairs of digits to separate.

## Algorithm Implementation and Development



### Problem a: Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image.



## Computational Results

### Problem a: Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image.



 

## Summary and Conclusions

