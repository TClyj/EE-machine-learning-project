# Analysis of Face Images Using Correlation Matrix and Singular Value Decomposition

Author: 

### Abstract: 

This report presents an analysis of a dataset of face images using correlation matrix and Singular Value Decomposition (SVD). The study investigates the relationships between images in the dataset, the most significant patterns in the data, and the efficiency of data representation using eigenvectors and SVD modes. The results demonstrate the effectiveness of SVD in capturing the underlying structure and patterns in the image data, as well as its potential for dimensionality reduction and feature extraction.

## Introduction and Overview

In this report, we analyze a dataset of face images consisting of 39 different faces with 65 lighting scenes for each face, resulting in a total of 2,414 images. The images are represented as a matrix X, where each image is downsampled to 32 x 32 pixels and converted to grayscale. Our analysis focuses on computing the correlation matrix between images, identifying the most highly correlated and uncorrelated image pairs, and applying Singular Value Decomposition (SVD) to find the principal component directions and the percentage of variance captured by each SVD mode.

## Theoretical Background

The analysis is based on the concepts of correlation matrix, eigenvectors, eigenvalues, and Singular Value Decomposition (SVD). The **correlation matrix** is a representation of the relationships between pairs of images in the dataset. 

`C_jk = Xj^T * Xk`

Each entry in the matrix (c_ij) represents the dot product between the i-th and j-th images in the matrix X, which is a measure of similarity between those images.

**Eigenvectors and eigenvalues** represent the most significant directions of variation in the data, eigenvectors and eigenvalues are properties of a square matrix, which in this case is the matrix `Y = X * X^T`

**Eigenvectors and eigenvalues** can be used to understand the underlying structure and patterns in the data. In the problem, computing the first six eigenvectors with the largest magnitude eigenvalues of the matrix Y can be used to represent the most significant directions of variation in the data. These directions can capture the most important features and patterns in the images.

**SVD** is a factorization of a matrix that can be used to analyze the underlying structure and patterns in the data, as well as for dimensionality reduction and feature extraction purposes. SVD decomposes a matrix X into three matrices: U, s (a diagonal matrix with singular values, usually represented as a 1D array), and V^T.

## Algorithm Implementation and Development

We implemented the algorithms for computing the correlation matrix, eigenvectors, eigenvalues, and SVD using Python and the NumPy and SciPy libraries. The implementation involves loading the dataset, computing the correlation matrix, sorting eigenvectors by descending eigenvalues, and performing SVD on the matrix X to find the principal component directions and the percentage of variance captured by each SVD mode.

### Problem a

After import the matrix X by using the provided code, we create the matrix C by using two for loop:
```Python
C = np.zeros((100, 100))
for j in range(100):
    for k in range(100):
        C[j, k] = np.dot(X[:, j].T, X[:, k])
```

Then plot the matrix:

```Python
plt.pcolor(C, cmap='viridis')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.show()
```

### Problem b

We set the diagonal elements of C to 19 to ignore them, as they represent the correlation of an image with itself, and use `np.argmax(C)` and `np.argmin(C)` to find the index of the maximum and minimum correlation value in the flattened version of the matrix C.
`np.unravel_index` is used to convert the flattened indices obtained from np.argmax and np.argmin back to their corresponding row and column indices in the original 100x100 matrix C. This allows us to identify the image pairs with the highest and lowest correlation values.

```Python
np.fill_diagonal(C, 10)  # Set diagonal elements to 10 to remove its influence on overall data set
max_idx = np.unravel_index(np.argmax(C), C.shape)
min_idx = np.unravel_index(np.argmin(C), C.shape)
```
Then plot the Images separately in two groups:

```Python
# Plot the most highly correlated images
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X[:, max_idx[0]].reshape(32, 32), cmap='gray')
ax1.set_title(f"Image {max_idx[0]}")
ax2.imshow(X[:, max_idx[1]].reshape(32, 32), cmap='gray')
ax2.set_title(f"Image {max_idx[1]}")
plt.suptitle("Most Highly Correlated Images")
plt.show()

# Plot the most uncorrelated images
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X[:, min_idx[0]].reshape(32, 32), cmap='gray')
ax1.set_title(f"Image {min_idx[0]}")
ax2.imshow(X[:, min_idx[1]].reshape(32, 32), cmap='gray')
ax2.set_title(f"Image {min_idx[1]}")
plt.suptitle("Most Uncorrelated Images")
plt.show()
```

### Problem c

In this problem we repeat what we did in part a but in a smaller range and provided index number, which is `[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]`.

```Python
# List of specified image indices
image_indices = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]

# Compute the 10x10 correlation matrix C
C = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        C[i, j] = np.dot(X[:, (image_indices[i] - 1)].T, X[:, (image_indices[j] - 1 )])
```
Then plot the new matrix C in pcolor:

```Python
plt.pcolor(C, cmap='viridis')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.xticks(range(10), [idx  for idx in image_indices])
plt.yticks(range(10), [idx  for idx in image_indices])
plt.show()
```

### Problem d

First we compute matrix Y and its eigenvalues and eigenvectors:

```Python
Y = np.dot(X, X.T)
eigenvalues, eigenvectors = np.linalg.eigh(Y)
```
Then we sort the eigenvalues and eigenvectors by the eigenvalue magnitudes in descending order and select the first six eigenvectors:

```Python
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

first_six_eigenvectors = eigenvectors[:, :6]
```
Then print it out:
```Python
print("First six eigenvectors with the largest magnitude eigenvalue:")
print(first_six_eigenvectors)
```

### Problem e & f

we perform SVD on the matrix X using `np.linalg.svd`. The SVD function returns three matrices: U, s, and Vt. The first six rows of the Vt matrix represent the first six principal component directions of the matrix X.
```Python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
principal_components = Vt[:6, :]

print("First six principal component directions:\n", principal_components)
```
Then we compute the norm of difference of absolute values of v1 from part d and u1 from part e.
```Python
u1 = U[:, 0]
v1 = eigenvectors[:, 0]

# Compute the norm of the difference of the absolute values of v1 and u1
norm_diff = np.linalg.norm(np.abs(v1) - np.abs(u1))
print("Norm of difference of their absolute values:", norm_diff)
```

### Problem g

We first sum up all the squared singular values to get the total variance in the dataset and then divide the square of each of the first six singular values by the total variance and multiply by 100 to get the percentage of variance captured by each SVD mode, and in the end print it out.
```Python
total_variance = np.sum(S ** 2)
percentage_variance = (S[:6] ** 2) / total_variance * 100
print("Percentage of variance captured by each of the first 6 SVD modes:")
for i, percentage in enumerate(percentage_variance):
    print(f"Mode {i + 1}: {percentage:.2f}%")

# Plot the first 6 SVD modes
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.imshow(U[:, i].reshape(32, 32), cmap='gray')
    ax.set_title(f"SVD Mode {i + 1}")
    ax.axis('off')
plt.suptitle("First 6 SVD Modes")
plt.show()
```

## Computational Results

### Problem a

Based on the process in development part, we get the following output:


From the image, we can determine the similarity between all these 100 images. In the graph, larger numbers represent a higher degree of correlation between images.

### Problem b

Based on the process in development part, we get the following output:

From the graph we can see that most correlated images are image 86 and 88, which match the location where brighter in the pcolor graph in problem a. Same for the uncorrelated images.

### Problem c

Based on the process in development part, we get the following output:

Although this graph is on a smaller scale than the one in part a, we can still see a similar data distribution, with the upper right having the brightest color.

### Problem d

Based on the process in development part, we get the following output:

### Problem e & f

### Problem g

## Summary and Conclusions

In conclusion, our analysis demonstrates the potential of correlation matrix and Singular Value Decomposition in understanding the relationships between images in a dataset, capturing the most important features and patterns, and representing the data more efficiently. This study highlights the importance of SVD in various applications, such as dimensionality reduction, feature extraction, and understanding the structure of the data.
