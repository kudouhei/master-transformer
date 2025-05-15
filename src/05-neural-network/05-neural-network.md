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

## 5. Training neural Networks
A feedforward neural net is an instance of supervised machine learning in which we know the correct output $y$ for each observation $x$.

Steps:
1. First, we’ll need a **loss function** that models the distance between the system output and the gold output, and it’s common to use the loss function used for logistic regression, the **cross-entropy loss**.
2. Second, to ﬁnd the parameters that minimize this loss function, we’ll use the **gradient descent** optimization algorithm
3. Third, gradient descent requires knowing the gradient of the loss function, the vector that contains the partial derivative of the loss function with respect to each of the parameters. 
   1. How do we partial out the loss over all those intermediate layers? The answer is the algorithm called **error backpropagation** or **backward differentiation**.

### 5.1 Loss function
**Cross-entropy**
The cross-entropy loss that is used in neural networks is the same one we saw for logistic regression.

If the neural network is being used as a **binary classifier**, with the sigmoid at the final layer, the loss function is the same logistic regression loss:

$$
L_{CE}(y, \hat{y}) = - \log p(y|x) = - [y \log \hat{y} + (1-y) \log (1-\hat{y})]
$$

The loss function for a single example **x** is the negative sum of the logs of the **K** output classes, each weighted by their probability $y_k$ in the gold distribution:

$$
L_{CE}(y, \hat{y}) = - \sum_{k=1}^{K} y_k \log \hat{y}_k
$$

**Negative log likelihood loss**
The cross-entropy loss is simply the negative log of the output probability corresponding to the correct class, and we therefore also call this the negative log likelihood loss:

$$
L_{CE}(y, \hat{y}) = - \log \hat{y}_c   
$$

where $c$ is the correct class.

Plugging in the softmax formula, and with $K$ the number of classes, we get:

$$
L_{CE}(y, \hat{y}) = - \log \frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}
$$

### 5.2 Gradient Descent
How do we compute the gradient of this loss function?

Computing the gradient requires the partial derivative of the loss function with respect to each parameter. For a network with one weight layer and sigmoid output, we could simply use the derivative of the loss that we used for logistic regression.

$$
\frac{\partial L_{CE} (y, \hat{y})}{\partial w_{j}} = (\hat{y} - y) x_j = (\sigma(w x + b) - y) x_j
$$

For a network with one weight layer and softmax output (=multinomial logistic regression), for a particular weight $w_{k}$ and input $x_i$, the partial derivative of the loss with respect to $w_{k}$ is:

$$
\frac{\partial L_{CE} (y, \hat{y})}{\partial w_{k,i}} = -(y_k - \hat{y}_k) x_{i} = -(y_k - p(y_k = 1 | x)) x_{i} = -(y_k - \frac{e^{w_k x + b_k}}{\sum_{j=1}^{K} e^{w_j x + b_j}}) x_{i}
$$

But these derivatives only give correct updates for one weight layer: the last one! For deep networks, computing the gradients for each weight is much more complex, since we are computing the derivative with respect to weight parameters that appear all the way back in the very early layers of the network, even though the loss is computed only at the very end of the network.

**Error backpropagation**
The solution to computing this gradient is an algorithm called error backpropagation or backprop. While backprop was invented specially for neural networks, it turns out to be the same as a more general procedure called backward differentiation, which depends on the notion of computation graphs.

### 5.3 Computation graphs
A computation graph is a representation of the process of computing a mathematical expression, in which the computation is broken down into separate operations, each of which is modeled as a node in a graph.

Consider computing the function $L(a,b,c) = c(a+2b)$. If we make each of the component addition and multiplication operations explicit, and add names (d and e) for the intermediate outputs, the resulting series of computations is:

$$
d = 2b
$$

$$
e = a + d
$$

$$
L = c \times e
$$

![computation graph](./images/08-forward.png)

In the forward pass of a computation graph, we apply each operation left to right, passing the outputs of each computation as the input to the next node.
