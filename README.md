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

### Problem 3: Do an analysis of the performance as a function of the time lag variable

### Problem 4: Do an analysis of the performance as a function of noise (add Gaussian noise to data)

### Problem 5: Do an analysis of the performance as a function of the number of sensors


## Computational Results

### Problem 1&2: Download the example code (and data) for sea-surface temperature which uses an LSTM/decoder and train the model and plot the results

### Problem 3: Do an analysis of the performance as a function of the time lag variable

### Problem 4: Do an analysis of the performance as a function of noise (add Gaussian noise to data)

### Problem 5: Do an analysis of the performance as a function of the number of sensors


## Summary and Conclusions

The study offers a systematic exploration of the factors that influence the performance of the SHRED model in time series prediction. By understanding the effect of different parameters such as time lags, noise levels, and the number of sensors, one can better configure and tune these models for specific tasks and environments. Future research could further optimize the SHRED model for noise resilience, explore the effect of other types of noise beyond Gaussian, and test the model's performance with varying types and numbers of sensors.
