# Analyzing Performance of Time Series Models Using SHRED

Author: Yijia Lu

### Abstract: 

This report presents a comprehensive analysis of the performance of time series models using the SHRED method, focusing on the impact of factors such as time lags, noise levels, and the number of sensors. The findings highlight the trade-offs in model performance and robustness in the face of noise and varying sensor inputs, and offer valuable insights for tuning SHRED models for optimum performance.

## Introduction and Overview

In this study, we leverage the SHRED (SHrinkage and Recurrent network Encoded Dynamics) method for time series modelling and perform an in-depth performance analysis. Our objective is to understand the impact of various parameters including time lags, the introduction of noise, and the number of sensors on the model performance.

## Theoretical Background

SHRED is a method that combines LSTM-based recurrent neural networks with a ridge regression penalty to forecast and reconstruct high-dimensional, multivariate time series. The analysis hinges upon the principle of using a dimensionality reduction technique and recurrent neural network to capture the dynamics of the system and make accurate predictions.

## Algorithm Implementation and Development

Example code for this project:

https://github.com/Jan-Williams/pyshred

### Problem 1&2: Download the example code (and data) for sea-surface temperature which uses an LSTM/decoder and train the model and plot the results

The training data we have consists of sequences of sensor readings. Each sequence is of length lags and contains readings from num_sensors different sensors. These sequences are used to train your SHRED model to predict the sensor readings at the next time step.

Specifically, the model is trained to predict the readings of the sensors at the locations specified by sensor_locations at the time step immediately after the end of each input sequence. So, for each input sequence, the corresponding output in the training data is a vector of sensor readings at these locations at the next time step.

Detailed output graph will be in result part.

### Problem 3: Do an analysis of the performance as a function of the time lag variable

Analyzing the performance of a model as a function of the time lag variable involves training and testing the model using different values of the time lag, then comparing the performance metrics for these different values.
```Python
# Define the list of lag values to test
lags = np.arange(1, 41, 10)

# Prepare a list to store the performance metrics for each lag value
performance_metrics = []

for lag in lags:
    # Redefine your datasets using the current lag value
    # You will need to adapt this to your specific code
    all_data_in = np.zeros((n - lag, lag, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lag, sensor_locations]
    
    # Create the model and train it on the training data
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=60, lr=1e-3, verbose=True, patience=5)
    
    # Evaluate the model on the test data
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    
    # Store the performance metric
    performance_metrics.append(error)
```

Then we plot the performance metrics as a function of the lag value. Detailed graph will be in result part.

### Problem 4: Do an analysis of the performance as a function of noise (add Gaussian noise to data)

The aim of this analysis is to understand how robust your model is to noise in the data. We will add Gaussian noise with different standard deviations to the data, retrain the model, and evaluate its performance each time.

```Python
# Define the list of standard deviations for the Gaussian noise
std_devs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

lags = 52

# Prepare a list to store the performance metrics for each level of noise
performance_metrics = []

for std_dev in std_devs:
    # Add Gaussian noise to the data
    noisy_X = load_X + np.random.normal(scale=std_dev, size=load_X.shape)
    
    # Preprocess the noisy data
    sc = MinMaxScaler()
    sc = sc.fit(noisy_X[train_indices])
    transformed_X = sc.transform(noisy_X)

    # Redefine your datasets using the noisy data
    # You will need to adapt this to your specific code
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
    
    # and so on for the rest of your dataset creation code...
    
    # Create the model and train it on the noisy data
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=60, lr=1e-3, verbose=True, patience=5)
    
    # Evaluate the model on the test data
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    
    # Store the performance metric
    performance_metrics.append(error)
```

Then we plot the performance metrics as a function of the standard deviation of the noise. Detailed graph will be in result part.

### Problem 5: Do an analysis of the performance as a function of the number of sensors

We're running an analysis to see how the performance of the SHRED model varies with the number of sensor inputs. To do this, we defined a function analyze_performance which encapsulates the entire process of data loading, preprocessing, model training, and performance evaluation.

We create an instance of the SHRED model, specifying num_sensors as the number of sensors the model should expect as input. Then train this model on the training data and evaluate it on the validation data.

After the model has been trained, we use it to reconstruct the test data, and compute the performance as the relative L2 norm of the difference between the reconstructed data and the ground truth data.

```Python
def analyze_performance(num_sensors):
    lags = 52
    load_X = load_data('SST')
    n = load_X.shape[0]
    m = load_X.shape[1]
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=60, lr=1e-3, verbose=True, patience=5)

    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    performance = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    return performance

# Define the range of number of sensors to analyze
num_sensors_range = range(1, 7)  # Example: 1 to 7 sensors

performance_results = []
for num_sensors in num_sensors_range:
    performance = analyze_performance(num_sensors)
    performance_results.append(performance)
```

We then plot the performance results in the result part.

## Computational Results

### Problem 1&2: Download the example code (and data) for sea-surface temperature which uses an LSTM/decoder and train the model and plot the results

![008b474ad6dddc59466df68fc811c11](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/b64ed056-566d-42e1-8e03-d3fbb8527f96)

The predictions line represents the sensor readings predicted by the model. Each point on this line is the reading predicted by the model for a specific sensor at a specific time.

Comparing these two lines gives you a visual sense of how well the model's predictions match the actual sensor readings. Ideally, the two lines should overlap or be very close to each other. Large discrepancies between the lines indicate times and sensors where the model's predictions were off.

### Problem 3: Do an analysis of the performance as a function of the time lag variable

![f4504869126bee5192eabf778b2d049](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/dc9efdd5-6e9b-4b9a-a303-614256b1a81c)

The graph will represent the performance of the model as a function of the time lag variable. The x-axis of the graph will represent different values of the time lag variable, while the y-axis will represent the model's performance metric (relative error in this case) associated with each value of the time lag.

### Problem 4: Do an analysis of the performance as a function of noise (add Gaussian noise to data)

![4c46d3a89f268b6d2327514e59081f5](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/3153081e-ec48-4bb3-bf89-eab8e72727b3)

As the standard deviation increase, the irregular error line start getting regular and error increase as the standard deviation increase. This is probably because as we add more noise, the true underlying pattern becomes harder to discern, and thus, it becomes more difficult for the model to make accurate predictions.

### Problem 5: Do an analysis of the performance as a function of the number of sensors

![4d3c6dc3ae1cf9c2aaada3bfe95ba4d](https://github.com/TClyj/EE-machine-learning-project/assets/126134377/a52cb78e-be64-447f-89dc-4616f6781f5f)

The performance decreases with more sensors, this might suggest that the model is being overwhelmed by the increased complexity and noise of the additional sensors.

## Summary and Conclusions

The study offers a systematic exploration of the factors that influence the performance of the SHRED model in time series prediction. By understanding the effect of different parameters such as time lags, noise levels, and the number of sensors, one can better configure and tune these models for specific tasks and environments. Future research could further optimize the SHRED model for noise resilience, explore the effect of other types of noise beyond Gaussian, and test the model's performance with varying types and numbers of sensors.
