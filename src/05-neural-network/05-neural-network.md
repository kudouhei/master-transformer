# Neural Network

## 1.Units
The building block of a neural network is a single computational unit. A unit takes a set of real valued numbers as input, performs some computation on them, and produces an output.

- **bias term**

    At its heart, a neural unit is taking a weighted sum of its inputs, with one additional term in the sum called a bias term. Given a set of inputs $x_1, ... x_n$, a unit has a set of corresponding weights $w_1, ... w_n$ and a bias $b$, so the weighted sum $z$ can be represented as:

    $$
    z = \sum_{i=1}^{n} w_i x_i + b
    $$

- **vector**  
  
    Often it’s more convenient to express this weighted sum using vector notation; Thus we’ll talk about z in terms of a weight vector w, a scalar bias b, and an input vector x, and we’ll replace the sum with the convenient **dot product**:

    $$
    z = w x + b
    $$

- **activation**

    The output of a unit is often passed to an activation function $f$, which is a function of the weighted sum. The activation function is a non-linear function that is applied to the weighted sum to produce the unit's output.

    $$
    y = a = f(z)
    $$

We’ll discuss **three popular non-linear functions $f$** below (the sigmoid, the tanh, and the rectified linear unit or ReLU)

- **sigmoid**

    $$
    y = \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

   ![sigmoid](./images/01-sigmoid.png)

    The sigmoid has a number of advantages; it maps the output into the range (0,1), which is useful in squashing outliers toward 0 or 1.

    Gives us the output of a neural unit:

    $$
    y = \sigma(w x + b) = \frac{1}{1 + e^{-(w x + b)}}
    $$

    ![sigmoid](./images/02-sigmoid.png)

- **tanh**

    The tanh function is a scaled version of the sigmoid, and it maps the output into the range (-1,1).

    $$
    y = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
    $$

- **ReLU**

    The ReLU function is a simple threshold function that outputs the input if it is positive, and 0 otherwise.

    $$
    y = \text{ReLU}(z) = \max(0, z)
    $$

    ![ReLU](./images/03-sigmoid.png)


## 2. The XOR problem

The XOR problem is a classic problem in machine learning. It is a binary classification problem where the input is a two-dimensional vector and the output is a binary label. The problem is that the XOR function is not linearly separable, meaning that no single line can separate the positive and negative examples.

![XOR](./images/04-xor.png)

## 3. Feedforward Neural Networks
A feedforward network is a multilayer network in which the units are connected with no cycles; the outputs from units in each layer are passed to units in the next higher layer, and no outputs are passed back to lower layers. (networks with cycles, called recurrent neural networks.)

**Multi-layer perceptrons (MLPs)**
For historical reasons multilayer networks, especially feedforward networks, are sometimes called multi-layer perceptrons (or MLPs)

Simple feedforward networks have three kinds of nodes: input units, hidden units, and output units.

![MLP](./images/05-feedforward.png)

The core of the neural network is the **hidden layer h** formed of hidden units $h_i$ , each of which is a neural unit.

**In fact, the computation only has three steps:**
1. Multiplying the weight matrix by the input vector $x$, 
2. Adding the bias vector $b$, 
3. Applying the activation function $g$

The output of the hidden layer, the vector $h$, is thus the following:

Eq.1
$$
h = \sigma(W x + b)
$$

Take a moment to convince yourself that the matrix multiplication in Eq.1  will compute the value of each $h_j$ as $\sigma ( \sum_{i=1}^{n} W_{ji} x_i + b_j )$ .

**Normalizing**
More generally for any vector $z$ of dimensionality $d$, the softmax is defined as:

Eq.2
$$
softmax(z_i)= \frac{e^{z_i}}{\sum_{j=1}^{d} e^{z_j}} \quad \text{for} \quad i = 1, ..., d
$$

Here are the final equations for a feedforward network with a single hidden layer, which takes an input vector $x$, outputs a probability distribution $y$, and is parameterized by weight matrices $W$ and $U$ and a bias vector $b$:


$$
h = \sigma(W x + b)
$$

$$
z = U h
$$

$$
y = softmax(z)
$$

## 4. Feedforward networks for NLP: Classiﬁcation

Let’s begin with a simple 2-layer sentiment classifier. 

$$
x = [x_1, x_2, ... x_N], \text{(each $x_i$ is a hand-designed feature)}
$$

$$
h = \sigma(Wx + b)
$$

$$
z = Uh
$$

$$
\hat{y} = softmax(z)
$$

![sentiment classifier](./images/06-feedforward.png)

**Pooling**
One simple baseline is to apply some sort of pooling function to the embeddings of all the words in the input.

For example, for a text with $n$ input words/tokens $w_1, ..., w_n$, we can turn the n embeddings $e(w_1), ..., e(w_n)$ (each of dimensionality $d$) into a single embedding also of dimensionality $d$ by just summing the embeddings, or by taking their mean (summing and then dividing by $n$):

$$
x_{mean} = \frac{1}{n} \sum_{i=1}^{n} e(w_i)
$$

The element-wise max of a set of $n$ vectors is a new vector whose $k$th element is the max of the $k$th elements of all the $n$ vectors.

$$
x = mean(e(w_1), e(w_2), ..., e(w_n))
$$

$$
h = \sigma(Wx + b)
$$

$$
z = Uh
$$

$$
\hat{y} = softmax(z)
$$

![sentiment classifier](./images/07-pooling.png)



