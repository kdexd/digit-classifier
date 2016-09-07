MNIST Handwritten Digit Classifier
==================================

An implementation of multilayer neural network using `numpy` library. The implementation 
is a modified version of Michael Nielsen's implementation in 
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) book. 


### Brief Background:

If you are familiar with basics of Neural Networks, feel free to skip this section. For 
total beginners who landed up here before reading anything about Neural Networks:

![Sigmoid Neuron](http://i.imgur.com/dOkT9Y9.png)

* Neural networks are made up of building blocks known as **Sigmoid Neurons**. These are 
named so because their output follows [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function).
* **x<sub>j</sub>** are inputs, which are weighted by **w<sub>j</sub>** weights and the 
neuron has its intrinsic bias **b**. The output of neuron is known as "activation ( **a** )".

_**Note:** There are other functions in use other than sigmoid, but this information for
now is sufficient for beginners._

* A neural network is made up by stacking layers of neurons, and is defined by the weights 
of connections and biases of neurons. Activations are a result dependent on a certain input.


### Why a modified implementation ?

This book and Stanford's Machine Learning Course by Prof. Andrew Ng are recommended as 
good resources for beginners. At times, it got confusing to me while referring both resources:

MATLAB has _1-indexed_ data structures, while `numpy` has them _0-indexed_. Some parameters 
of a neural network are not defined for the input layer, so there was a little mess up in 
mathematical equations of book, and indices in code. For example according to the book, the 
bias vector of second layer of neural network was referred as `bias[0]` as input layer (first 
layer) has no bias vector. I found it a bit inconvenient to play with.

I am fond of Scikit Learn's API style, hence my class has a similar structure of code. While 
theoretically it resembles the book and Stanford's course, you can find simple methods such 
as `fit`, `predict`, `validate` to train, test, validate the model respectively.


### Naming and Indexing Convention:

I have followed a particular convention in indexing quantities.
Dimensions of quantities are listed according to this figure.

![Small Labelled Neural Network](http://i.imgur.com/HdfentB.png)


#### **Layers**
* Input layer is the **0<sup>th</sup>** layer, and output layer 
is the **L<sup>th</sup>** layer. Number of layers: **N<sub>L</sub> = L + 1**.
```
sizes = [2, 3, 1]
```

#### **Weights**
* Weights in this neural network implementation are a list of 
matrices (`numpy.ndarrays`). `weights[l]` is a matrix of weights entering the 
**l<sup>th</sup>** layer of the network (Denoted as **w<sup>l</sup>**).  
* An element of this matrix is denoted as **w<sup>l</sup><sub>jk</sub>**. It is a 
part of **j<sup>th</sup>** row, which is a collection of all weights entering 
**j<sup>th</sup>** neuron, from all neurons (0 to k) of **(l-1)<sup>th</sup>** layer.  
* No weights enter the input layer, hence `weights[0]` is redundant, and further it 
follows as `weights[1]` being the collection of weights entering layer 1 and so on.
```
weights = |¯   [[]],    [[a, b],    [[p],   ¯|
          |              [c, d],     [q],    |
          |_             [e, f]],    [r]]   _|
```

#### **Biases**
* Biases in this neural network implementation are a list of one-dimensional 
vectors (`numpy.ndarrays`). `biases[l]` is a vector of biases of neurons in the 
**l<sup>th</sup>** layer of network (Denoted as **b<sup>l</sup>**).  
* An element of this vector is denoted as **b<sup>l</sup><sub>j</sub>**. It is a 
part of **j<sup>th</sup>** row, the bias of **j<sup>th</sup>** in layer.  
* Input layer has no biases, hence `biases[0]` is redundant, and further it 
follows as `biases[1]` being the biases of neurons of layer 1 and so on.
```
biases = |¯   [[],    [[0],    [[0]]   ¯|
         |     []],    [1],             |
         |_            [2]],           _|
```

#### **'Z's**
* For input vector **x** to a layer **l**, **z** is defined as: 
**z<sup>l</sup> = w<sup>l</sup> . x + b<sup>l</sup>**
* Input layer provides **x** vector as input to layer 1, and itself has no input, 
weight or bias, hence `zs[0]` is redundant.
* Dimensions of `zs` will be same as `biases`.

#### **Activations**
* Activations of **l<sup>th</sup>** layer are outputs from neurons of **l<sup>th</sup>** 
which serve as input to **(l+1)<sup>th</sup>** layer. The dimensions of `biases`, `zs` and 
`activations` are similar.
* Input layer provides **x** vector as input to layer 1, hence `activations[0]` can be related 
to **x** - the input training example.
