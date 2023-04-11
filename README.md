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

![ad36c5913e4099f886f21005bf8f56e](https://user-images.githubusercontent.com/126134377/231071622-a0c4d021-a876-4aac-8bdb-471aebdc2a55.png)

By applying the LSE and provided model with given data sets, we can see that we get the possible parameter values that fit in this model from our code and the red line match the trend of data sets, which means LSE works as our expection.

For question two, we then present the results of generating the 2D error landscape and analyze the number of minima found. 
Because there are 6 possible combination for 4 variables, so we output 6 graphs in pcolor in the order of AB fix, AC fix, AD fix, BC fix, BD fix, CD fix. 

      ![c51982374c3659268cd9832b0738c9a](https://user-images.githubusercontent.com/126134377/231072574-8d3dddce-3862-4c3a-a9d0-ed949e826fcc.png)
![ff2559a083223374de6282c089b3d72](https://user-images.githubusercontent.com/126134377/231072697-ac6df2e7-d6da-4206-962a-3ba2676de813.png)
![af749c4154060093c369567f95494ab](https://user-images.githubusercontent.com/126134377/231072734-ca6fa781-4f3d-4361-8ae6-ab1579d4e5f7.png)
![23214a80331b41ff9b7639dd182be15](https://user-images.githubusercontent.com/126134377/231072770-ad48cc08-fd15-4710-95d4-f45936612b68.png)
![87db3dce220fb5bc3cd48e8fabf284c](https://user-images.githubusercontent.com/126134377/231072782-65eac8ce-0762-4b9d-9532-66588d5a4b31.png)
![8ead8b4856171932ffb3c6690070ce1](https://user-images.githubusercontent.com/126134377/231072791-f412b90a-c28e-425c-a594-8561dc529511.png)

In the 2D loss (error) landscape, the color intensity represents the magnitude of the loss function, where darker colors indicate higher losses and lighter colors indicate lower losses. In this question the dark color regions will represent the minima for different fix situation. By comparing the area of the dark regions, we can see that while A and B is not fixed, our result will be very unstable. For the parameter that link to higher degree variable, more influence will appear if we change its value during our training process.
     
Finally, in question three and four, we compare the least-square error results for fitting a line, parabola, and 19th degree polynomial over training and test data sets.

This is the error number and graph for question three:
* Training errors
* Line: 2.24
* Parabola: 2.13
* 19th degree polynomial: 0.03
* Test errors
* Line: 3.36
* Parabola: 8.71
* 19th degree polynomial: 28617752784.43

![6546f3a6f6fe4360738b05fb9e4e3cf](https://user-images.githubusercontent.com/126134377/231072821-78483e66-cde0-4fbe-8982-f846ca51d8a5.png)

Since we are using np.polyfit to find the best fitting line, when the degree is too high, the result will be pretty extreme and poorly conditioned. So here we can see that in the situation of 19th degree polynomial the line fit the training data well but have pretty large error in test data sets. Other than that, line and parabola fit in the training data (error is low) but can't fit in test data very well. The larger degree the larger error.

This is the error number and graph for question four:
* Training errors
* Line: 1.81
* Parabola: 1.81
* 19th degree polynomial: 0.22
* Test errors
* Line: 2.95
* Parabola: 2.94
* 19th degree polynomial: 81.93

![6af07f13d84cc886640f4f4fce38de4](https://user-images.githubusercontent.com/126134377/231072845-3b693c89-4d76-4ef9-a021-e8fc792eea91.png)

Similar to what we get in question three, the 19th degree polynomial is extreme. However, in question four we change our training data and find that the overall error on test data in smaller than what in previous question. The error for 19th degree polynomial is much smaller due to its extreme property. By comparing, we can see that with a better initial guess (training data), our output solution will be better and less error.

Also, one intresting things about question four is the error for line and parabola is almost same in training data sets, which means they are almost overlap with each other if they are in the same graph, so I split them into two graphs.

## Summary and Conclusions:
We investigated the use of least-squares error for modeling and optimizing parameters in Python.
We presented an algorithm for implementing least-squares error and demonstrated its application on a sample data set.
Using the results of the least-squares error model, we generated a 2D error landscape and analyzed the number of minima found.
Finally, we compared the least-square error results for fitting a line, parabola, and 19th degree polynomial over different training and test data sets.
Our results highlight the usefulness of least-squares error for modeling and optimizing parameters in a wide range of applications.
