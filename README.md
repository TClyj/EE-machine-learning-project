# Modeling and Parameter Optimization using Least-Squares Error

### Author: Yijia Lu

In this homework, we investigate the use of least-squares error for modeling and optimizing parameters in Jupyter Notebook (Python).
We begin by introducing the theoretical background and mathematical concepts underlying least-squares error. 
Then we present an algorithm for implementing least-squares error in Python and demonstrate its application on a sample data set. 
Finally, we analyze the results of the least-squares error model and highlight the usefulness of least-squares error for modeling and optimizing parameters in a wide range of applications.

## Introduction and Overview:
The least-squares error is a powerful technique for modeling and optimizing parameters for a given data set.
This technique is widely used in a variety of fields, including machine learning, statistics, and data analysis.

In this homework, we investigate the use of least-squares error to model and optimize parameters for a given data set. 
We present an algorithm for implementing least-squares error in Python and demonstrate its application on a sample data set.
Using the results of the least-squares error model, we generate a 2D error landscape by fixing two parameters and sweeping through the other two parameters and visualize the error landscape using pcolor.
Finally, we compare the least-square error results for fitting a line, parabola, and 19th degree polynomial over training and test data sets.

## Theoretical Background:
In class, we discuss about the least-squares error and explaining its importance in modeling and parameter optimization. 
Least-squares error (LSE) is a method used to estimate the parameters of a mathematical model by minimizing the sum of the squares of the differences between the predicted and actual values.

For example, let's say you have a set of data points and you want to fit a line to them. LSE would involve finding the line that minimizes the sum of the squares of the distances between each data point and the line. This method can also be used with other types of models, such as quadratic or exponential functions.
![332443970cccf19aca30bbe05a0b90a](https://user-images.githubusercontent.com/126134377/231067499-e36be68a-07f1-48d3-954d-68f866bd4c74.png)

LSE is commonly used in regression analysis, where the goal is to find the line or curve that best fits the data. By minimizing the sum of the squares of the errors, LSE helps to ensure that the resulting model fits the data as closely as possible.

## Algorithm Implementation and Development:

### Detail outputs image and data will be show in the next section.

Inside the file we present an algorithm for implementing least-squares error in Python. 
First we import several useful library before we start working on problem one.
* import numpy as np
* import scipy.optimize as opt
* from scipy.optimize import curve_fit
* import matplotlib.pyplot as plt

Then we define the given model in a function:
* def f(x, A, B, C, D):
*   return A * np.cos(B*x) + C*x + D

and apply our least square error in python by using:
* np.sqrt(np.sum((f(X, *popt) - Y)**2) / len(X))

In this case, we can find our minimum error using LSE by find the parameters ABCD through curve fitting the provided data points.

For question two, we fix two of the parameters (A and B for example) and sweep through values of the other two parameters (C and D if A and B are fixed) using np.linspace. 
We then create a grid of parameter values using np.meshgrid and compute the error for each combination of parameter values using nested loops. 
Finally, we visualize the results in a 2D color plot using plt.pcolor.

For question three, we first need to split the data into training and test sets. Then using np.polyfit we can fing the line, parabola and 19th degree polynomial
that fit in the training model. Finally we can find the error for training and test data sets by applying least square error.

For question four, the process is similar to question three but changing the training and test data sets.

## Computational Results:
In this section, we present the computational results of our analysis. 
For question one, we first discuss the results of the least-squares error model applied to the sample data set. 
For question two, we then present the results of generating the 2D error landscape and analyze the number of minima found. 
Finally, in question three and four, we compare the least-square error results for fitting a line, parabola, and 19th degree polynomial over training and test data sets.

## Summary and Conclusions:
We investigated the use of least-squares error for modeling and optimizing parameters in Python.
We presented an algorithm for implementing least-squares error and demonstrated its application on a sample data set.
Using the results of the least-squares error model, we generated a 2D error landscape and analyzed the number of minima found.
Finally, we compared the least-square error results for fitting a line, parabola, and 19th degree polynomial over training and test data sets.
Our results highlight the usefulness of least-squares error for modeling and optimizing parameters in a wide range of applications.
