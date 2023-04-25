# Analysis of the MNIST Dataset: SVD, LDA, SVM, and Decision Tree Classifiers

Author: Yijia Lu

### Abstract: 

In this report, we analyze the MNIST dataset of handwritten digits using various techniques, such as Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees. We explore the underlying structure of the data, build classifiers to identify individual digits, and compare the performance of these classifiers on both the training and test sets.

## Introduction and Overview

The MNIST dataset is a popular dataset containing 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels, representing a single digit. The goal of this analysis is to explore the dataset and build classifiers to identify individual digits accurately. We will use various techniques such as SVD, LDA, SVM, and Decision Trees and compare their performance.

## Theoretical Background

**SVD** is a factorization of a matrix that can be used to analyze the underlying structure and patterns in the data, as well as for dimensionality reduction and feature extraction purposes. SVD decomposes a matrix X into three matrices: U, s (a diagonal matrix with singular values, usually represented as a 1D array), and V^T.

**LDA** is a linear classification algorithm that projects data points onto a lower-dimensional space to maximize the separation between different classes. It assumes that the data is normally distributed and that the covariances for each class are identical. LDA works by finding the linear combination of features that best separates the classes.

In the questions, LDA was used to classify pairs and triplets of digits from the MNIST dataset. By training the LDA classifier on the filtered training data (consisting of only the selected digits), we were able to find the linear decision boundary that best separates the chosen digits. The accuracy of the LDA classifier was then evaluated on the test data.

**SVM** is a powerful classification algorithm that finds the optimal hyperplane separating the data into different classes. The algorithm aims to maximize the margin between the hyperplane and the closest data points from each class, known as support vectors. SVM can handle both linear and non-linear classification tasks using kernel functions.

In the questions, SVM was applied to classify all ten digits in the MNIST dataset. The SVM classifier was trained on the entire training data, and its accuracy was evaluated on the test data. A comparison between LDA, SVM, and Decision Trees was then performed on the hardest and easiest pairs of digits to separate.

**Decision Trees** are a class of algorithms that recursively split the data into subsets based on feature values. They create a tree-like structure, where each internal node represents a decision based on a feature, and each leaf node represents the predicted class label. Decision Trees are simple to understand, can handle both categorical and continuous features, and are resistant to noise.

In the questions, the Decision Tree classifier was applied to classify all ten digits in the MNIST dataset. Similar to the SVM classifier, the Decision Tree was trained on the entire training data, and its accuracy was evaluated on the test data. The performance of the Decision Tree classifier was then compared to LDA and SVM on the hardest and easiest pairs of digits to separate.

## Algorithm Implementation and Development

Firstly, we used TensorFlow's Keras API to easily load and preprocess the MNIST dataset, making it suitable for our analysis and classification tasks.

```Python
! pip install tensorflow
```
Also import some useful library like scikit-learn, which can provided us with a wide range of algorithms, such as LDA, SVM, and Decision Trees. It also offered tools for evaluating classifier performance, such as accuracy_score and train_test_split functions.
```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
```

### Problem a: Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image.

To perform an SVD analysis of the MNIST dataset, we will start by loading the dataset. Then, we will reshape the images and create a data matrix. Finally, we will perform SVD on the data matrix.
```Python
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape each image into a column vector
train_images_flat = train_images.reshape(train_images.shape[0], -1).T
test_images_flat = test_images.reshape(test_images.shape[0], -1).T

# Perform SVD on the data matrix
u, s, vh = np.linalg.svd(train_images_flat, full_matrices=False)
```
The train_images_flat array has a shape of (784, 60000), where each column represents an image from the training set, and the test_images_flat array has a shape of (784, 10000), where each column represents an image from the testing set. These arrays are transposed (.T) so that the images are represented as columns.

Then we Visualize the top singular vectors (eigendigits) as images
```Python
n_components = 10
plt.figure(figsize=(12, 6))
for i in range(n_components):
    plt.subplot(2, 5, i+1)
    plt.imshow(u[:, i].reshape(28, 28), cmap='gray')
    plt.title(f'Eigendigit {i+1}')
    plt.axis('off')
plt.show()
```
Result image will be in result part.

### Problem b: What does the singular value spectrum look like and how many modes are necessary for good image reconstruction? (i.e. what is the rank r of the digit space?)

To analyze the singular value spectrum and determine the number of modes necessary for a good image reconstruction, we can plot the singular values and analyze the cumulative explained variance. We also find the number of modes necessary for a good image reconstruction. Here, we use 90% explained variance as the threshold for good image reconstruction.
```Python
# Plot the singular value spectrum
plt.figure(figsize=(10, 5))
plt.plot(s, linewidth=2)
plt.xlabel('Mode')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')
plt.grid()
plt.show()

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(s**2) / np.sum(s**2)

# Find the number of modes necessary for a good image reconstruction
# Here, we use 90% explained variance as the threshold for good image reconstruction
r = np.argmax(cumulative_explained_variance >= 0.9) + 1

# Plot the cumulative explained variance
plt.figure(figsize=(10, 5))
plt.plot(cumulative_explained_variance, linewidth=2)
plt.xlabel('Number of Modes')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Modes')
plt.axvline(x=r, color='r', linestyle='--', label=f'Rank r = {r}')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Explained Variance')
plt.legend()
plt.grid()
plt.show()

print(f'Rank r of the digit space (90% explained variance): {r}')
```
Result images will be in result part.

### Problem c: What is the interpretation of the U, Σ, and V matrices?

In the context of Singular Value Decomposition (SVD), a given matrix A is factorized into three matrices: U, Σ, and V*. Here's a brief interpretation of each matrix:
* **U**: The U matrix contains the left singular vectors of the original matrix A. In the case of the MNIST dataset, these vectors are also known as "eigendigits" when reshaped into 28x28 images.

* **Σ**: The Σ matrix is a diagonal matrix containing the singular values of A. These singular values are non-negative and arranged in descending order, with the largest singular value in the top-left corner. The singular values represent the scaling factors for each mode (corresponding to the columns of U) and indicate the importance of each mode in capturing the variance in the dataset.

* **V***: The V* matrix (the conjugate transpose of V) contains the right singular vectors of the original matrix A. The columns of V* represent the orthogonal basis vectors that span the row space of A. 

The SVD factorization helps us analyze the structure of the dataset, reduce dimensionality, remove noise, and extract significant features for further analysis or machine learning applications. To better understand these topic, we will plot some graph based on the data we have.
```Python
# Plot singular value spectrum
plt.figure()
plt.plot(s)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')

# Reconstruct some images using the first few singular values and modes
n_modes = [1, 10, 50]
image_idx = 2

fig, axs = plt.subplots(1, len(n_modes) + 1, figsize=(10, 3))
axs[0].imshow(train_images[image_idx], cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original Image')

for i, mode in enumerate(n_modes):
    reconstruction = u[:, :mode] @ np.diag(s[:mode]) @ vh[:mode, :]
    reconstructed_image = reconstruction[:, image_idx].reshape(28, 28)
    axs[i + 1].imshow(reconstructed_image, cmap='gray')
    axs[i + 1].axis('off')
    axs[i + 1].set_title(f'{mode} Modes')

plt.suptitle('Image Reconstructions Using Different Numbers of Modes')
plt.tight_layout()
plt.show()
```
Result images will be in result part.

### Problem d: On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For example, columns 2,3, and 5.

This code loads the MNIST dataset, normalizes the images, reshapes them into column vectors, and performs SVD on the data matrix. Then, it projects the dataset onto three selected V-modes (columns 2, 3, and 5) and creates a 3D plot with each point colored according to its digit label. 

For each digit, create a boolean mask where `train_labels` equals the current digit. Use the mask to filter the `projection` data for the current digit and plot the corresponding 3D points using `ax.scatter()`. Each digit's points will be plotted in a different color.
```Python
from mpl_toolkits.mplot3d import Axes3D

selected_modes = [1, 2, 4]  # Columns 2, 3, and 5 (0-indexed)
projection = np.dot(train_images_flat.T, vh[:, selected_modes])

# Create a 3D plot colored by digit labels
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for digit in range(10):
    mask = train_labels == digit
    ax.scatter(projection[mask, 0], projection[mask, 1], projection[mask, 2], label=digit)

ax.set_xlabel(f'V-Mode {selected_modes[0] + 1}')
ax.set_ylabel(f'V-Mode {selected_modes[1] + 1}')
ax.set_zlabel(f'V-Mode {selected_modes[2] + 1}')
ax.set_title('Projection onto Selected V-Modes')
ax.legend()
plt.show()
```
Result images will be in result part.

### Problem e: Pick two digits. See if you can build a linear classifier (LDA) that can reasonable identify/classify them.

To build a linear classifier using Linear Discriminant Analysis (LDA) to identify and classify two selected digits from the MNIST dataset, we choose two digits to classify, and create a mask to filter the training and testing datasets.

This code will train a linear classifier using LDA to identify and classify two chosen digits from the MNIST dataset. The classifier is trained on the filtered training dataset and tested on the filtered testing dataset. The accuracy of the classifier is then calculated and displayed, providing a measure of how well the LDA classifier can distinguish between the two selected digits.

```Python
# Choose two digits to classify, and create a mask to filter the training and testing datasets:
digit1, digit2 = 2, 6

train_mask = (train_labels == digit1) | (train_labels == digit2)
test_mask = (test_labels == digit1) | (test_labels == digit2)

train_images_filtered = train_images[train_mask]
train_labels_filtered = train_labels[train_mask]
test_images_filtered = test_images[test_mask]
test_labels_filtered = test_labels[test_mask]
```
Then we reshape the filtered images into column vectors, train the LDA classifier on the filtered training dataset and test the trained LDA classifier on the filtered testing dataset, and calculate the accuracy:
```Python
train_images_flat = train_images_filtered.reshape(train_images_filtered.shape[0], -1)
test_images_flat = test_images_filtered.reshape(test_images_filtered.shape[0], -1)

lda = LinearDiscriminantAnalysis()
lda.fit(train_images_flat, train_labels_filtered)

predictions = lda.predict(test_images_flat)
accuracy = accuracy_score(test_labels_filtered, predictions)

print(f"Accuracy of the LDA classifier on digits {digit1} and {digit2}: {accuracy * 100:.2f}%")
```
Then we plot a graph including LDA Projection and Decision Boundary for Digits we choose(Which is 2 and 6 here).
```Python
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
train_images_flat_scaled = scaler.fit_transform(train_images_flat)
test_images_flat_scaled = scaler.transform(test_images_flat)

# Perform LDA on the scaled data
X_train_lda = lda.fit_transform(train_images_flat_scaled, train_labels_filtered)
X_test_lda = lda.transform(test_images_flat_scaled)

# Calculate the accuracy on the test set
lda_accuracy = lda.score(test_images_flat_scaled, test_labels_filtered)
print(f"LDA classifier accuracy on digits {digit1} and {digit2}: {lda_accuracy * 100:.2f}%")

# Plot the dataset projected onto the first LDA component with the decision boundary
plt.figure()
colors = ['b', 'r']
for digit, color in zip([digit1, digit2], colors):
    plt.scatter(X_train_lda[train_labels_filtered == digit, 0], np.zeros(np.sum(train_labels_filtered == digit)), c=color, alpha=0.5, label=f"Digit {digit}")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('LDA Component')
plt.title(f'LDA Projection and Decision Boundary for Digits {digit1} and {digit2}')
plt.legend()
plt.show()
```
This code snippet first scales the data using `StandardScaler` and then fits the LDA classifier on the scaled data. The LDA components are calculated, and a scatter plot is created, showing the dataset projected onto the first LDA component (which is the only component in this case since LDA reduces the dimensionality to one less than the number of classes).

Result image will be in result part.

### Problem f: Pick three digits. Try to build a linear classifier to identify these three now.
The Process of this problem is similar to previous question, the only difference is three digits are choosed to build the linear classifier. 

This code will train a linear classifier using LDA to identify and classify three chosen digits from the MNIST dataset. The classifier is trained on the filtered training dataset and tested on the filtered testing dataset. The accuracy of the classifier is then calculated and displayed, providing a measure of how well the LDA classifier can distinguish between the three selected digits.

```Python
digit1, digit2, digit3 = 2, 6, 7

train_mask = (train_labels == digit1) | (train_labels == digit2) | (train_labels == digit3)
test_mask = (test_labels == digit1) | (test_labels == digit2) | (test_labels == digit3)

train_images_filtered = train_images[train_mask]
train_labels_filtered = train_labels[train_mask]
test_images_filtered = test_images[test_mask]
test_labels_filtered = test_labels[test_mask]

# Reshape the filtered images into column vectors:
train_images_flat = train_images_filtered.reshape(train_images_filtered.shape[0], -1)
test_images_flat = test_images_filtered.reshape(test_images_filtered.shape[0], -1)

# Train the LDA classifier on the filtered training dataset:
lda = LinearDiscriminantAnalysis()
lda.fit(train_images_flat, train_labels_filtered)

# Test the trained LDA classifier on the filtered testing dataset, and calculate the accuracy:
predictions = lda.predict(test_images_flat)
accuracy = accuracy_score(test_labels_filtered, predictions)

print(f"Accuracy of the LDA classifier on digits {digit1}, {digit2}, and {digit3}: {accuracy * 100:.2f}%")
```
Then modify the data to create a graph, code is similar to what in previous question, so I don't show the code here.

Result image will be in result part.


### Problem g: Which two digits in the data set appear to be the most difficult to separate? Quantify the accuracy of the separation with LDA on the test data.

To determine which two digits in the MNIST dataset are the most difficult to separate using an LDA classifier, we can evaluate the accuracy of the LDA classifier for all possible pairs of digits and find the pair with the lowest accuracy.

First we define a function to calculate the LDA classification accuracy for a given pair of digits, this function can also been used in next question.
```Python
def lda_accuracy(digit1, digit2, train_images, train_labels, test_images, test_labels):
    train_mask = (train_labels == digit1) | (train_labels == digit2)
    test_mask = (test_labels == digit1) | (test_labels == digit2)

    train_images_filtered = train_images[train_mask]
    train_labels_filtered = train_labels[train_mask]
    test_images_filtered = test_images[test_mask]
    test_labels_filtered = test_labels[test_mask]

    train_images_flat = train_images_filtered.reshape(train_images_filtered.shape[0], -1)
    test_images_flat = test_images_filtered.reshape(test_images_filtered.shape[0], -1)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_images_flat, train_labels_filtered)

    predictions = lda.predict(test_images_flat)
    accuracy = accuracy_score(test_labels_filtered, predictions)

    return accuracy
```

Then we iterate through all possible digit pairs, calculate the LDA classification accuracy and find the pair with the lowest accuracy:
```Python
min_accuracy = 1.0
min_digit_pair = None

for i in range(10):
    for j in range(i + 1, 10):
        accuracy = lda_accuracy(i, j, train_images, train_labels, test_images, test_labels)
        if accuracy < min_accuracy:
            min_accuracy = accuracy
            min_digit_pair = (i, j)

print(f"The most difficult digits to separate are {min_digit_pair[0]} and {min_digit_pair[1]}, with an accuracy of {min_accuracy * 100:.2f}%.")
```
Then we write some code to display both question g and h's graph, detailed code will be in next part and result images will be in result part

### Problem h: Which two digits in the data set are most easy to separate? Quantify the accuracy of the separation with LDA on the test data.

Like what I said in problem g, we will use the same defined function but a different loop. We iterate through all possible digit pairs, calculate the LDA classification accuracy and find the pair with the highest accuracy:
```Python
max_accuracy = 0.0
max_digit_pair = None

for i in range(10):
    for j in range(i + 1, 10):
        accuracy = lda_accuracy(i, j, train_images, train_labels, test_images, test_labels)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_digit_pair = (i, j)

print(f"The easiest digits to separate are {max_digit_pair[0]} and {max_digit_pair[1]}, with an accuracy of {max_accuracy * 100:.2f}%.")
```
In the below code, we use the `plot_lda_boundary()` function to visualize the LDA decision boundary for both the hardest and easiest pairs of digits. The function takes the training and testing images and labels, the two digits to classify, and a title for the plot.
```Python
def plot_lda_boundary(train_images, train_labels, test_images, test_labels, digit1, digit2, title):
    mask_train = (train_labels == digit1) | (train_labels == digit2)
    mask_test = (test_labels == digit1) | (test_labels == digit2)

    train_images_filtered = train_images[mask_train]
    train_labels_filtered = train_labels[mask_train]
    test_images_filtered = test_images[mask_test]
    test_labels_filtered = test_labels[mask_test]

    train_images_flat = train_images_filtered.reshape(train_images_filtered.shape[0], -1)
    test_images_flat = test_images_filtered.reshape(test_images_filtered.shape[0], -1)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_images_flat, train_labels_filtered)

    transformed_data = lda.transform(test_images_flat)

    plt.figure()
    plt.scatter(transformed_data[test_labels_filtered == digit1], [0] * len(transformed_data[test_labels_filtered == digit1]), label=f"Digit {digit1}")
    plt.scatter(transformed_data[test_labels_filtered == digit2], [0] * len(transformed_data[test_labels_filtered == digit2]), label=f"Digit {digit2}")
    plt.axvline(x=0, color='red', linestyle='--', label='LDA Decision Boundary')
    plt.xlabel('LDA Component')
    plt.title(title)
    plt.legend()
    plt.show()

plot_lda_boundary(train_images, train_labels, test_images, test_labels, hardest_digits[0], hardest_digits[1], f"Hardest Pair: {hardest_digits[0]} and {hardest_digits[1]}")
plot_lda_boundary(train_images, train_labels, test_images, test_labels, easiest_digits[0], easiest_digits[1], f"Easiest Pair: {easiest_digits[0]} and {easiest_digits[1]}")
```

Result images will be in result part

### Problem i: SVM (support vector machines) and decision tree classifiers were the state-of-the-art until about 2014. How well do these separate between all ten digits?

To evaluate the performance of SVM (Support Vector Machines) and Decision Tree classifiers on the MNIST dataset for all ten digits, we can train and test these classifiers on the dataset and compute their accuracy.
```Python
# Train SVM classifier on training data
svm = svm.SVC()
svm.fit(train_images_flat.T, train_labels)

# Calculate accuracy of SVM classifier on test data
# Use SVM classifier to predict labels on test data
svm_predictions = svm.predict(test_images_flat.T)
svm_accuracy = accuracy_score(test_labels, svm_predictions)

print(f"SVM classifier accuracy on all ten digits: {svm_accuracy * 100:.2f}%")
```
Similar to SVM, we do the same thing for decision tree:
```Python
# Train and test a Decision Tree classifier on the dataset:
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_images_flat.T, train_labels)

dt_predictions = decision_tree.predict(test_images_flat.T)
dt_accuracy = accuracy_score(test_labels, dt_predictions)

print(f"Decision Tree classifier accuracy on all ten digits: {dt_accuracy * 100:.2f}%")
```
Then we draw two confusion Matrix to display the seperation of SVM and Decision Tree Classifier.
```Python
# Compute confusion matrices for both classifiers
svm_confusion_matrix = confusion_matrix(test_labels, svm_predictions)
dt_confusion_matrix = confusion_matrix(test_labels, dt_predictions)

# Plot confusion matrices as heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(svm_confusion_matrix, annot=True, fmt='d', ax=ax1)
ax1.set_title('SVM Classifier Confusion Matrix')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

sns.heatmap(dt_confusion_matrix, annot=True, fmt='d', ax=ax2)
ax2.set_title('Decision Tree Classifier Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

plt.show()
```
Result image will be in result part.

### Problem j: Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above).

First, let's create a function to evaluate the performance of each classifier on a given pair of digits:
```Python
def evaluate_classifiers(digit1, digit2, train_images, train_labels, test_images, test_labels):
    train_mask = (train_labels == digit1) | (train_labels == digit2)
    test_mask = (test_labels == digit1) | (test_labels == digit2)

    train_images_filtered = train_images[train_mask]
    train_labels_filtered = train_labels[train_mask]
    test_images_filtered = test_images[test_mask]
    test_labels_filtered = test_labels[test_mask]

    train_images_flat = train_images_filtered.reshape(train_images_filtered.shape[0], -1)
    test_images_flat = test_images_filtered.reshape(test_images_filtered.shape[0], -1)

    # Create classifiers
    lda = LinearDiscriminantAnalysis()
    svm = SVC()
    dt = DecisionTreeClassifier()

    # Train classifiers
    lda.fit(train_images_flat, train_labels_filtered)
    svm.fit(train_images_flat, train_labels_filtered)
    dt.fit(train_images_flat, train_labels_filtered)

    # Test classifiers
    lda_predictions = lda.predict(test_images_flat)
    svm_predictions = svm.predict(test_images_flat)
    dt_predictions = dt.predict(test_images_flat)

    # Calculate accuracies
    lda_accuracy = accuracy_score(test_labels_filtered, lda_predictions)
    svm_accuracy = accuracy_score(test_labels_filtered, svm_predictions)
    dt_accuracy = accuracy_score(test_labels_filtered, dt_predictions)

    return lda_accuracy, svm_accuracy, dt_accuracy
```
Then evaluate the performance of the classifiers on the hardest and easiest pairs of digits to separate:
```Python
# Hardest pair of digits to separate
hardest_digits = min_digit_pair
lda_accuracy, svm_accuracy, dt_accuracy = evaluate_classifiers(*hardest_digits, train_images, train_labels, test_images, test_labels)

print(f"Classifier performance on hardest pair of digits to separate ({hardest_digits[0]} and {hardest_digits[1]}):")
print(f"LDA accuracy: {lda_accuracy * 100:.2f}%")
print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")
print(f"Decision Tree accuracy: {dt_accuracy * 100:.2f}%\n")

# Easiest pair of digits to separate
easiest_digits = max_digit_pair
lda_accuracy, svm_accuracy, dt_accuracy = evaluate_classifiers(*easiest_digits, train_images, train_labels, test_images, test_labels)

print(f"Classifier performance on easiest pair of digits to separate ({easiest_digits[0]} and {easiest_digits[1]}):")
print(f"LDA accuracy: {lda_accuracy * 100:.2f}%")
print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")
print(f"Decision Tree accuracy: {dt_accuracy * 100:.2f}%")
```
This code will evaluate the performance of the LDA, SVM, and Decision Tree classifiers on both the hardest and easiest pairs of digits to separate, and print out their respective accuracies.

Then we create a bar chart to visualize the performance of each classifier on the hardest and easiest pairs of digits to separate.

Result image will be in result parts.

## Computational Results

### Problem a: Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image.

![e597ac09a0b970d6e70d3f8b364ea38](https://user-images.githubusercontent.com/126134377/234246214-0a6ae9ae-dace-4de3-b3d7-66cdfb9a5cef.png)
By visualizing the top 10 eigendigits, we can gain insight into the underlying structure of the data and the most important patterns that distinguish the handwritten digits. 


### Problem b: What does the singular value spectrum look like and how many modes are necessary for good image reconstruction? (i.e. what is the rank r of the digit space?)

`Rank r of the digit space (90% explained variance): 53`

The singular value spectrum refers to the distribution of singular values obtained from Singular Value Decomposition (SVD). 
![2eedca60f16eb2d149fac03ccb5aa4d](https://user-images.githubusercontent.com/126134377/234246691-7e6f3798-7006-449f-89b1-964fa7c8d3f3.png)

In the singular value spectrum, each singular value corresponds to a "mode." The modes represent the importance of each eigendigit in capturing the variance in the dataset. The higher the singular value, the more significant the corresponding eigendigit is in describing the dataset's structure.

### Problem c: What is the interpretation of the U, Σ, and V matrices?

![64546533119d548e1f1fd06580c3911](https://user-images.githubusercontent.com/126134377/234249079-6634429a-ca82-4634-865b-2cb72310f3be.png)
First plots the singular value spectrum, which shows the magnitude of each singular value in the Σ matrix. Then, it demonstrates how the original images can be reconstructed using different numbers of modes by combining the corresponding columns in U, singular values in Σ, and rows in V. The reconstructions become more accurate as more modes are used, indicating that the SVD captures the essential structure of the data.

### Problem d: On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For example, columns 2,3, and 5.

![4455e5cd5cc08d31f275455116705fb](https://user-images.githubusercontent.com/126134377/234252931-adae3f15-af13-40c7-8906-9c079f5cdcaa.png)

In the 3D plot, the three axes represent the projections of the MNIST images onto the selected V-modes (columns of the V matrix). The axes correspond to the coefficients of the images when represented in the reduced-dimensionality space spanned by these three V-modes. Each point in the 3D plot represents an image from the MNIST dataset projected onto this reduced-dimensionality space. The x, y, and z coordinates of each point indicate the contribution of the selected V-modes (in this case, V-mode 2, V-mode 3, and V-mode 5) to that specific image.

The points in the 3D plot are colored according to their digit labels (0 to 9). Each color represents a different digit, and the points of the same color belong to the same digit class. By visualizing the data in this way, we can observe how well the selected V-modes separate the different digit classes in the reduced-dimensionality space. If the points of the same color (digit) are clustered together, it suggests that the selected V-modes can capture the variance within the digit class and discriminate between different digit classes. This information can be useful for tasks such as image recognition, dimensionality reduction, and feature extraction.


### Problem e: Pick two digits. See if you can build a linear classifier (LDA) that can reasonable identify/classify them.

`Accuracy of the LDA classifier on digits 2 and 6: 97.89%` 
![17beffb1563d21e7c749812eec39e24](https://user-images.githubusercontent.com/126134377/234257040-6d5bf289-c8e9-4ea9-a670-73ea4fb361d0.png)

The decision boundary is visualized as a horizontal dashed line at y=0, and The plot shows that the LDA classifier can reasonably separate the two digits (2 and 6), as the points for each digit are mostly located on opposite sides of the decision boundary.

### Problem f: Pick three digits. Try to build a linear classifier to identify these three now.

`LDA classifier accuracy on digits 2, 6, and 7: 96.72%`
![2f5abc83736cff89663bb9f4684af15](https://user-images.githubusercontent.com/126134377/234257621-edb32e94-4414-4b0f-b737-3acc507abf49.png)

The plot shows that the LDA classifier can reasonably separate the three digits (2, 6, and 7). However, LDA doesn't always produce a linear boundary between classes, so the separation might not be perfect.


### Problem g & h: Which two digits in the data set appear to be the most difficult to separate? Quantify the accuracy of the separation with LDA on the test data. Which two digits in the data set are most easy to separate? Quantify the accuracy of the separation with LDA on the test data.

* `The most difficult digits to separate are 5 and 8, with an accuracy of 95.12%.`
* `The easiest digits to separate are 6 and 9, with an accuracy of 99.59%.`
![f6ef01d55fcfb3c0b4f7a415d0e3eb9](https://user-images.githubusercontent.com/126134377/234264860-9c1baea7-60dc-48f6-a182-87465e515d15.png)

In the above code, we use the plot_lda_boundary() function to visualize the LDA decision boundary for both the hardest and easiest pairs of digits. The function takes the training and testing images and labels, the two digits to classify, and a title for the plot.

The resulting plots show the LDA-transformed data for each digit as points along the horizontal axis, with the LDA decision boundary (the red dashed line) separating the two classes. The easiest pair of digits will have a clearer separation between the classes, while the hardest pair will have more overlap between the classes.


### Problem i: SVM (support vector machines) and decision tree classifiers were the state-of-the-art until about 2014. How well do these separate between all ten digits?
* `SVM classifier accuracy on all ten digits: 97.92%`
* `Decision Tree classifier accuracy on all ten digits: 87.83%`
![5748ad28b505b6ad3097609f61bac31](https://user-images.githubusercontent.com/126134377/234272296-44c1a271-8250-4487-89c1-64ac358f6d0e.png)
A confusion matrix is a table that is often used to describe the performance of a classification model on a set of data for which the true values are known. The heatmap visualizes the confusion matrix, showing the number of correctly and incorrectly classified images for each digit. The darker the color, the higher the count of images in that cell of the matrix.

The diagonal of the confusion matrix represents the correctly classified images, while the off-diagonal elements indicate the misclassifications. By analyzing the heatmaps, we can see that like the data we get, the SVM will have denser data seperation, which means SVM can provide more accurate classifier on all ten digits.

### Problem j: Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above).
* Classifier performance on hardest pair of digits to separate (5 and 8):
* LDA accuracy: 95.12%
* SVM accuracy: 99.57%
* Decision Tree accuracy: 96.20%

* Classifier performance on easiest pair of digits to separate (6 and 9):
* LDA accuracy: 99.59%
* SVM accuracy: 99.90%
* Decision Tree accuracy: 99.64%

![5c74b8eb5e8138ed95846ae43a9366c](https://user-images.githubusercontent.com/126134377/234267695-0d342011-b852-4936-b5ff-0b71c997165b.png)

From the data and graph above we can see that SVM will provide more accurate classifier performance compare to what in LDA and Decision Tree. 

## Summary and Conclusions

This study focuses on the analysis of the MNIST dataset and the application of various classification algorithms to identify handwritten digits. The initial analysis involves performing Singular Value Decomposition (SVD) on the dataset and projecting the data into PCA space. By visualizing the top singular vectors as eigendigits, we gain insights into the inherent structure of the data.

Next, we apply Linear Discriminant Analysis (LDA) to classify pairs and triplets of digits. The LDA classifier is trained on filtered training data and its accuracy is evaluated on test data. Through this process, we identify the hardest and easiest pairs of digits to separate using LDA.

We then explore the performance of Support Vector Machines (SVM) and Decision Tree classifiers on the entire dataset, comprising all ten digits. The classifiers are trained on the complete training data and their accuracies are evaluated on the test data. The performance of LDA, SVM, and Decision Trees is compared on the hardest and easiest pairs of digits to separate, providing a comprehensive understanding of their capabilities in the context of the MNIST dataset.

In conclusion, this study demonstrates the application of various classification algorithms to the MNIST dataset and provides valuable insights into the performance of LDA, SVM, and Decision Tree classifiers. The results highlight the strengths and weaknesses of these classifiers and offer a deeper understanding of their suitability for different classification tasks within the domain of handwritten digit recognition.
