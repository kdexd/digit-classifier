MNIST Handwritten Digit Classifier
==================================

An implementation of multilayer neural network using Python's `numpy` library. 
The implementation is a modified version of Michael Nielsen's implementation 
in [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) book. 


### Why a modified implementation ?

This book and Stanford's Machine Learning Course by Prof. Andrew Ng are recommended as 
good resources for beginners. One problem I faced during going through both resources was:

> Stanford Course uses MATLAB, which has _1-indexed_ vectors and matrices.  
> The book uses `numpy` library of Python, which has _0-indexed_ vectors and arrays.

Further more, some parameters of a neural network are not defined for the input layer, 
hence I didn't get a hang of implementation using Python.  
For example according to the book, the bias vector of second layer of neural network 
was referred as `bias[0]` as input layer(first layer) has no bias vector. So indexing 
got weird with `numpy` and MATLAB.


### My Naming and Indexing Convention:
 _to be done !_