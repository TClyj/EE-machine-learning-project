# Analyzing and Modeling Data Using a Three-Layer Feed-Forward Neural Network and Comparing Model Performance

Author: Yijia Lu

### Abstract: 

This study aims to evaluate the performance of a three-layer feed-forward neural network on a given time series dataset and compare the effectiveness of various machine learning models, including feed-forward neural networks, Long Short-Term Memory (LSTM) networks, Support Vector Machines (SVM), and Decision Trees, on the MNIST image dataset. The neural network was trained using different portions of the time series data, and the mean squared error (MSE) was calculated for the training and test sets to assess the model's performance. Furthermore, the study involved training and evaluating the mentioned machine learning models on the PCA-transformed MNIST dataset and comparing their accuracy using a bar chart.

The results provide insights into the impact of data splits on the neural network's performance and the relative strengths and weaknesses of the various classifiers for the given tasks.

## Introduction and Overview

In this report, we analyze a dataset containing 31 data points and fit a three-layer feed-forward neural network to the data. We train and evaluate the neural network model using different portions of the dataset and compare the performance of the model to other classifiers such as LSTM, SVM, and Decision Trees.

## Theoretical Background

A ***Feed-Forward Neural Network (FFNN)*** is a type of artificial neural network where information flows from input to output without any cycles or loops. It consists of layers of neurons (also called nodes), including an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to all neurons in the previous and next layers, with associated weights and biases. The network learns to map input data to the corresponding output through a process called backpropagation, which adjusts the weights and biases by minimizing a loss function.

***LSTM*** is a type of Recurrent Neural Network (RNN) architecture designed to learn and model sequential data. LSTMs have memory cells and three gates (input, forget, and output gates) that control the flow of information inside the cell. This architecture allows LSTMs to capture long-range dependencies and mitigate the vanishing gradient problem.

***Support Vector Machines (SVM)*** are supervised learning algorithms primarily used for classification and regression tasks. The main idea behind SVM is to find a hyperplane that best separates the data into different classes. The optimal hyperplane maximizes the margin, which is the distance between the hyperplane and the nearest data points (called support vectors) from each class.

***Decision Trees*** are a type of supervised learning algorithm used for classification and regression tasks. They work by recursively splitting the input data into subsets based on specific feature values, resulting in a tree-like structure. The tree's leaves represent the final decision, while the internal nodes represent the features used for splitting the data.


## Algorithm Implementation and Development

The data is first preprocessed by reshaping and scaling it to a suitable format for the neural network model. The scaling process ensures that the input data is within a manageable range for the model, which is essential for obtaining optimal performance.

### Problem 1(i): Reconsider the data from homework one: (i) Fit the data to a three layer feed forward neural network.
```Python
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
To fit data into a neural network model, the GPT provided me a library called "tensorflow", which have developed methods to define, compile, and train a feed-forward neural network. The model is trained for 500 epochs with a batch size of 4 using the Adam optimizer and mean squared error as the loss function. The data is split into training and testing sets, and the model's performance is evaluated on the test set.

```Python
# Reshape and scale the data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=1))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, Y_train, epochs=500, batch_size=4, verbose=0)

# Evaluate the model on the test set
loss = model.evaluate(X_test, Y_test)
```

Also, I do the same process but not using the tensorflow library and their result accuracy is difference. Detailed graph will be showed in Results part.

### Problem 1(ii): Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.

This part we modified the data splitting part to use the first 20 data points for training and the remaining 10 data points for testing. It then trains the three-layer feed-forward neural network using the training data and computes the mean squared error for both the training and test sets.

Split the data into training and testing sets
```Python
X_train, X_test = X_scaled[:20], X_scaled[20:]
Y_train, Y_test = Y_scaled[:20], Y_scaled[20:]
```
Then train the model and compute the mse on both training and testing sets.
```Python
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, Y_train, epochs=500, batch_size=4, verbose=0)

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
mse_train = tf.keras.losses.mean_squared_error(Y_train, Y_train_pred).numpy()
mse_test = tf.keras.losses.mean_squared_error(Y_test, Y_test_pred).numpy()
```

### Problem 1(iii): Repeat (ii) but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to (ii)

Here we do probably the same thing with the previous but a different split value:
```Python
X_train = np.vstack((X_scaled[:10], X_scaled[-10:]))
Y_train = np.vstack((Y_scaled[:10], Y_scaled[-10:]))
X_test = X_scaled[10:20]
Y_test = Y_scaled[10:20]
```

Then we use the mse_tain and mse_test in problem (ii) and (iii) for doing a comparation in a line chart.
```Python
x_train = np.arange(len(mse_train_scenario1))
x_test = np.arange(len(mse_test_scenario1))

x_train2 = np.arange(len(mse_train_scenario2))
x_test2 = np.arange(len(mse_test_scenario2))

plt.figure(figsize=(12, 6))

plt.plot(x_train, mse_train_scenario1, label='Scenario 1 - Training Data', linestyle='-', marker='o')
plt.plot(x_test, mse_test_scenario1, label='Scenario 1 - Test Data', linestyle='-', marker='x')

plt.plot(x_train2, mse_train_scenario2, label='Scenario 2 - Training Data', linestyle='--', marker='o')
plt.plot(x_test2, mse_test_scenario2, label='Scenario 2 - Test Data', linestyle='--', marker='x')

plt.xlabel('Data Points')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Mean Squared Errors for Two Scenarios')

plt.legend()
plt.grid()
plt.show()
```

Output graph will showed in Results part.

### Problem 1（iv) Compare the models fit in homework one to the neural networks in (ii) and (iii)

We will compare the error between HW1 and HW4 in Results part.

### Problem 2(i): Now train a feedforward neural network on the MNIST data set. You will start by performing the following analysis: (i) Compute the first 20 PCA modes of the digit images.

This part is similar to what we did in previous homework. We load the MNIST dataset, flatten the images and compute the PCA modes
```Python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

x_all = np.vstack((x_train_flat, x_test_flat))

n_components = 20
pca = PCA(n_components=n_components)
pca.fit(x_all)

# Transform the data using the first 20 PCA modes
x_all_pca = pca.transform(x_all)
```
By printing the shape in the end, we can check if we get the correct data output.

### Problem 2(ii) Build a feed-forward neural network to classify the digits. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

For the FFNN training, the work process is similar to what we did in problem 1 but with a different dataset. 

For the LSTM training, the tensorflow library can also provide existing model for us to use
```Python
from tensorflow.keras.layers import Dense, LSTM, Reshape
```
Then we define and train the model similar to FFNN:
```Python
# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(Reshape((n_components, 1), input_shape=(n_components,)))
model_lstm.add(LSTM(128, activation='tanh'))
model_lstm.add(Dense(10, activation='softmax'))

# Compile and train the model
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(x_train_pca, y_train_cat, epochs=20, batch_size=128, verbose=1)

# Evaluate the LSTM model
_, lstm_accuracy = model_lstm.evaluate(x_test_pca, y_test_cat)
```
For SVM and Decision Tree classifier, we can just do what we do in past HW:
```Python
# Train and evaluate the SVM classifier
svm_classifier = SVC(kernel='rbf', gamma='scale')
svm_classifier.fit(x_train_pca, y_train)
svm_accuracy = accuracy_score(y_test, svm_classifier.predict(x_test_pca))

# Train and evaluate the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train_pca, y_train)
dt_accuracy = accuracy_score(y_test, dt_classifier.predict(x_test_pca))
```

In the end, we print the accuracy of four method in a bar chart and compare their accuracy.
```Python
model_names = ['Feed-Forward NN', 'LSTM', 'SVM', 'Decision Tree']
accuracies = [ffnn_accuracy, lstm_accuracy, svm_accuracy, dt_accuracy]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0.8, 1)

# Add accuracy values above the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', fontsize=12)

plt.show()
```


## Computational Results

### Problem 1(i): Reconsider the data from homework one: (i) Fit the data to a three layer feed forward neural network.

**Test loss**: 0.011351206339895725
![be24a255ea2c9819116a9994530e8c6](https://user-images.githubusercontent.com/126134377/237008989-77d6a4a2-3f76-4f9a-a6b9-5d85bda5a210.png)

I also did the same question but not using exist tensorflow library
![f62f1d8f7219774d49a2be66d871005](https://user-images.githubusercontent.com/126134377/237009745-5cd3df8d-93e9-44f5-a0ab-99e7db1136aa.png)

By compare the two graph, we can see that by using the existing library, the output graph is more accurate. So we will keep using the library for our neural network training in future problems.

### Problem 1(ii): Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.

![d2194923d1a20fa57294fed86972603](https://user-images.githubusercontent.com/126134377/237010701-d62daf76-3a57-498d-8be9-ed69c6433e7a.png)

We will use the above data points in next question and create a graph to do the comparation.

### Problem 1(iii): Repeat (ii) but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to (ii)

![d16a54a096f8e0e9920d0450722b7ea](https://user-images.githubusercontent.com/126134377/237012063-417137e2-efc3-4438-914c-c4fc3f7e5b22.png)

Next is the chart for comparing:
![89052b3a5096ec3bebd6a8e0822a4d3](https://user-images.githubusercontent.com/126134377/237012451-d1165c19-fa0c-4618-8dd4-23396e92bf42.png)

Here we can see that using FFNN the error on training data on the two data split is similar, But the test error on dataset2 (which is first and least 10 data for training) is smaller than what in dataset1 (which is first 20 data points). 

### Problem 1（iv) Compare the models fit in homework one to the neural networks in (ii) and (iii)

The following two graphs are from HW1:

![27caabffc94eef9ac7075cbeab27bfb](https://user-images.githubusercontent.com/126134377/237013805-f2740cb2-4e1c-4a1b-831e-75aa6ed5866e.png)
![e9ea9fd5de9e33367776ff71a24d04c](https://user-images.githubusercontent.com/126134377/237013820-133b8180-2582-4dd5-8ca0-8b2d710d6ece.png)

We can see that the rule we found in previous problem is also work here. Picking first and least 10 data points can provide more accurate model. 

Also, by comparing the error data in HW1 with error list in HW4. We can see that FFNN can provide less error model compare to what we did in HW1.

### Problem 2(i): Now train a feedforward neural network on the MNIST data set. You will start by performing the following analysis: (i) Compute the first 20 PCA modes of the digit images.

Here we just print out the shape of the PCA modes to check we get the correct MNIST dataset.

**Transformed data shape**: (70000, 20)

### Problem 2(ii) Build a feed-forward neural network to classify the digits. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

**Accuracy**:

Feed-Forward Neural Network: 0.9564

LSTM: 0.8988

SVM: 0.9754

Decision Tree: 0.8478

![254126489391addcdff1bac943c1442](https://user-images.githubusercontent.com/126134377/237016217-6730bb03-1cb8-4370-b821-b849f2bf3e2f.png)

Here we can clearly see the difference between the accuracy between each model. From the graph we can see that SVM have the most accuracy with the provided dataset. 

## Summary and Conclusions

Through the process of training and evaluating the neural network model using different portions of the dataset, we gain insight into the model's performance and the impact of data splits. Additionally, by comparing the neural network model to other classifiers, we can assess the relative strengths and weaknesses of each method. This analysis can inform the choice of classifier for specific tasks and guide further optimization of model architectures and hyperparameters.
