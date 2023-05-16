# Comparative Study of FFNN, RNN, LSTM, and ESN for Predicting the Dynamics of the Lorenz System

Author: Yijia Lu

### Abstract: 

This report explores and compares the efficiency of four types of neural network models - Feedforward Neural Network (FFNN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Echo State Network (ESN) - in predicting the dynamics of the Lorenz system, a complex nonlinear system.

## Introduction and Overview

The Lorenz system, a set of three differential equations, is known for its chaotic behavior and is often used as a benchmark for testing prediction models. In this study, we generate data from the Lorenz system and apply four types of neural networks to predict the system's dynamics. The aim is to compare their performances and identify the most efficient model.

## Theoretical Background

The ***Lorenz system*** is a deterministic, three-dimensional system defined by three nonlinear differential equations. These equations describe the rate of change of three variables in response to three parameters. The system exhibits a range of behaviors as the parameters vary, including stable points, limit cycles, and chaotic behavior.


## Algorithm Implementation and Development

For the Lorenz equations (code given out previously in class emails), consider the following.

### Problem 1: Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well your NN works for future state prediction for ρ = 17 and ρ = 35.

First we modify the provided Lorenz equations by changing rho to three values and get our training dataset. 

```Python
dt = 0.01
T = 8
t = np.arange(0, T+dt, dt)
beta = 8/3
sigma = 10
rho = [10, 28, 40]  # values of rho to use for training
```
Also modify the Generation process of different rho:
```Python
# Generate trajectories for each value of rho
for i, rho in enumerate(rho):
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho)) for x0_j in x0])
    
    for j in range(100):
        start_idx = (i * 100 + j) * (len(t) - 1)
        end_idx = (i * 100 + j + 1) * (len(t) - 1)
        nn_input[start_idx:end_idx, :] = x_t[j, :-1, :]
        nn_output[start_idx:end_idx, :] = x_t[j, 1:, :]
        
        x, y, z = x_t[j,:,:].T
        ax.plot(x, y, z, linewidth=1)
        ax.scatter(x0[j,0], x0[j,1], x0[j,2], color='r')
```

Then we train a FFNN model by using the scaled ***nn_input*** and ***nn_output***:

```Python
# train the model
history = model.fit(nn_input_scaled, nn_output_scaled, epochs=10, batch_size=32, validation_split=0.2)
```
With trained FFNN model, we will use the two testing rho, which is 17 and 35, to get the predicted values base on our model and compare it with the real values to check the accuracy.

```Python
# Make predictions
nn_output_pred_scaled = model.predict(nn_input_test_scaled)
nn_output_pred = scaler.inverse_transform(nn_output_pred_scaled)

# Compute the error
error = nn_output_test - nn_output_pred
mse = np.mean(error**2)
print(f"Mean squared error: {mse}")
```
Detailed result graph and Mean squared error will be showed in result part.


### Problem 2: Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.

In this question we implemented each of the four types of neural networks using Python and trained them on data generated from the Lorenz system. The training data consisted of trajectories generated with three different values of the parameter rho. The test data consisted of trajectories generated with two different values of rho.

***FFNN:***
Training part: 

```Python
model_ffnn = Sequential([
    Dense(10, input_shape=(3,), activation='relu'),
    Dense(3)
])

model_ffnn.compile(optimizer='adam', loss='mse')
history_ffnn = model_ffnn.fit(nn_input_scaled, nn_output_scaled, epochs=10, batch_size=32, validation_split=0.2)
```

Testing part:
```Python
nn_output_pred_scaled_ffnn = model_ffnn.predict(nn_input_test_scaled)
nn_output_pred_ffnn = scaler.inverse_transform(nn_output_pred_scaled_ffnn)

error_ffnn = nn_output_test - nn_output_pred_ffnn
mse_ffnn = np.mean(error_ffnn**2)
print(f"FFNN mean squared error: {mse_ffnn}")
```

***RNN:***
Training part: 

```Python
# Reshape the input for RNN
nn_input_rnn = nn_input_scaled.reshape((nn_input_scaled.shape[0], 1, nn_input_scaled.shape[1]))

# Define the RNN model architecture
model_rnn = tf.keras.Sequential()
model_rnn.add(tf.keras.layers.SimpleRNN(10, input_shape=(1, 3), activation='relu'))
model_rnn.add(tf.keras.layers.Dense(3))

# Compile the RNN model with Adam optimizer and mean squared error loss
model_rnn.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE)

# Train the RNN model on the normalized input and output data
history_rnn = model_rnn.fit(nn_input_rnn, nn_output_scaled, epochs=10, batch_size=32, validation_split=0.2)
```

Testing part:
```Python
# Reshape the input for RNN
nn_input_test_rnn = nn_input_test_scaled.reshape((nn_input_test_scaled.shape[0], 1, nn_input_test_scaled.shape[1]))

# Make predictions on the normalized test data using the trained RNN model
nn_output_pred_scaled_rnn = model_rnn.predict(nn_input_test_rnn)

# Rescale the predicted output back to the original scale using the scaler
nn_output_pred_rnn = scaler.inverse_transform(nn_output_pred_scaled_rnn)

# Compute the error between the predicted and actual test data
error_rnn = nn_output_test - nn_output_pred_rnn
mse_rnn = np.mean(error_rnn**2)

# Print the mean squared error
print(f"Mean squared error (RNN): {mse_rnn}")
```

***LSTM:***
Training part: 

```Python
# Reshape the input for LSTM
nn_input_lstm = nn_input_scaled.reshape((nn_input_scaled.shape[0], 1, nn_input_scaled.shape[1]))

# Define the LSTM model architecture
model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.LSTM(10, input_shape=(1, 3), activation='relu'))
model_lstm.add(tf.keras.layers.Dense(3))

# Compile the LSTM model with Adam optimizer and mean squared error loss
model_lstm.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE)

# Train the LSTM model on the normalized input and output data
history_lstm = model_lstm.fit(nn_input_lstm, nn_output_scaled, epochs=10, batch_size=32, validation_split=0.2)
```

Testing part:
```Python
# Reshape the input for LSTM
nn_input_test_lstm = nn_input_test_scaled.reshape((nn_input_test_scaled.shape[0], 1, nn_input_test_scaled.shape[1]))

# Make predictions on the normalized test data using the trained LSTM model
nn_output_pred_scaled_lstm = model_lstm.predict(nn_input_test_lstm)

# Rescale the predicted output back to the original scale using the scaler
nn_output_pred_lstm = scaler.inverse_transform(nn_output_pred_scaled_lstm)

# Compute the error between the predicted and actual test data
error_lstm = nn_output_test - nn_output_pred_lstm
mse_lstm = np.mean(error_lstm**2)

# Print the mean squared error
print(f"Mean squared error (LSTM): {mse_lstm}")
```

***ESN:***
For this model, we use the ESM classes writen by other user, the class I use is from: 
https://github.com/cknd/pyESN.git

By using the ESN class, we can use ***.fit*** and ***.predict*** to train and test the datasets.
```Python
# Initialize the ESN
esn = ESN(n_inputs=3, 
          n_outputs=3, 
          n_reservoir=200, 
          spectral_radius=0.95, 
          sparsity=0.0, 
          noise=0.001)

# Train the ESN
pred_train = esn.fit(nn_input_scaled, nn_output_scaled)

# Test the ESN
pred_test = esn.predict(nn_input_test_scaled)

# Truncate nn_output_scaled to match the length of pred_train and pred_test
truncated_output_scaled = nn_output_scaled[-len(pred_train):]
truncated_output_test_scaled = nn_output_test_scaled[-len(pred_test):]

# Compute the Mean Squared Error (MSE)
mse_esn_train = np.mean((pred_train - truncated_output_scaled)**2)
mse_esn_test = np.mean((pred_test - truncated_output_test_scaled)**2)

print(f'Training MSE for ESN: {mse_esn_train}')
print(f'Testing MSE for ESN: {mse_esn_test}')
```

Detailed MSE values will be listed in Result part. 

## Computational Results

### Problem 1: Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well your NN works for future state prediction for ρ = 17 and ρ = 35.

This is the output graph of training model of FFNN: 

![47e8dce323ef44724bf287e634e102f](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/9f92611a-c32e-4562-a918-093af3e25db2)

Here we can see that as more training epoch, the loss (MSE) will be less and model will be more accurate.

Next graph showed how well our predict data fit in the real data:

![98b1c5e948a05f59ace70bd409a58b2](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/5a6da4da-1c8a-4a5a-8065-1aa749e72177)

Here we can see the MSE of our predict model is pretty small which means our training model is accurate.


### Problem 2: Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.

*FFNN mean squared error: 0.0711

*RNN Mean squared error: 0.0513

*LSTM Mean squared error: 0.0181

*ESN Mean squared error: 0.300

And we plot the number in a bar chart to better compare these values.

![fe3f970f72f88aa141d5e707c3a5048](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/e46fa99d-1093-4edc-944b-d31ec2edf4ea)

From the graph we can see that base on the provided training and testing datasets, it is evident that the LSTM model has the lowest MSE, while the ESN model exhibits the highest error.


## Summary and Conclusions

Through the process of training and evaluating the neural network model using different portions of the dataset, we gain insight into the model's performance and the impact of data splits. Additionally, by comparing the neural network model to other classifiers, we can assess the relative strengths and weaknesses of each method. This analysis can inform the choice of classifier for specific tasks and guide further optimization of model architectures and hyperparameters.

