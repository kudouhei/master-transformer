import nltk
import numpy as np
from src.utils import get_batches
from src.utils import compute_pca
from src.utils import get_dict
import re
from matplotlib import pyplot as plt

# create corpus for training
with open('./data/hamlet.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# clean data by removing punctuation, tokenizing by words, and converting to lower case
data = re.sub(r"[,!?;-]", ".", data)  # replace punctuation with '.'
data = nltk.word_tokenize(data)
data = [ch.lower() for ch in data if ch.isalpha() or ch == '.'] 

print("Number of tokens:", len(data), "\n", data[500:515])

# get Bag of Words (BoW) features
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary:", len(fdist))
print("Most Frequent Tokens:", fdist.most_common(20))

# create 2 dictionaries: word2idx and idx2word
word2Ind, Ind2word = get_dict(data)  # word2idx: word -> index, idx2word: index -> word

V = len(word2Ind)
print("Vocabulary size: ", V, "\n")

# create neural network with 1 layer and 2 parameters
# 随机初始化模型参数（权重和偏置），N是隐藏层维度，V是词汇表大小。
def initialize_model(N, V, random_seed=1):
    """
    Inputs:
        N: dimension of hidden vector
        V: dimension of vocabulary
        random_seed: seed for consistent results in tests
    Outputs:
        W1, W2, b1, b2: initialized weights and biases
    """
    np.random.seed(random_seed)

    W1 = np.random.randn(N, V) # weight matrix for hidden layer
    W2 = np.random.randn(V, N) # weight matrix for output layer
    b1 = np.random.rand(N, 1)  # bias for hidden layer
    b2 = np.random.rand(V, 1)  # bias for output layer

    return W1, W2, b1, b2


# Create our final classification layer, which makes all possibilities add up to 1
def softmax(z):
    """
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    """
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    return yhat

# Define the behavior for moving forward through our model, along with an activation function
def forward_prop(x, W1, W2, b1, b2):
    """
    Inputs:
        x: average one-hot vector for the context
        W1,W2,b1,b2: weights and biases to be learned
        W₁: 输入层到隐藏层的权重矩阵（形状 (N, V))
        x: 输入词的平均one-hot向量 (形状 (V, 1))   
        b₁: 隐藏层的偏置（形状 (N, 1))
        W₂: 隐藏层到输出层的权重矩阵（形状 (V, N))
        b₂: 输出层的偏置（形状 (V, 1))
    Outputs:
        z: output score vector
    """
    h = W1 @ x + b1
    h = np.maximum(0, h)  # 只有激活值>0的神经元才会传递梯度
    z = W2 @ h + b2
    return z, h

# Define how we determine the distance between ground truth and model predictions
def compute_cost(y, yhat, batch_size):
    """
    Inputs:
        y: ground truth
        yhat: prediction
        batch_size: size of the batch
    """
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(np.log(1-yhat), 1-y)

    cost = -1 / batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)

    return cost

# Define how we move backward through the model and collect gradients
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    """
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  weights and biases
        batch_size: batch size
     Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of weights and biases
    """
    l1 = np.dot(W2.T, yhat - y)
    l1 = np.maximum(0, l1)

    grad_W1 = np.dot(l1, x.T) / batch_size
    grad_W2 = np.dot(yhat - y, h.T) / batch_size
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size

    return grad_W1, grad_W2, grad_b1, grad_b2


# put it all together and train the model
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    """
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        num_iters: number of iterations
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases
    """
    W1, W2, b1, b2 = initialize_model(N, V)

    batch_size = 128
    iters = 0
    C = 2  
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = compute_cost(y, yhat, batch_size)
        
        if (iters +1 )% 10 ==0:
            print(f"iters: {iters+1} cost: {cost:.6f}")
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        W1 = W1 - alpha * grad_W1
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2


# Train the model
C = 2
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)

# After listing 2.4 is done and gradient descent has been executed
words = [
    "king",
    "queen",
    "lord",
    "man",
    "woman",
    "prince",
    "ophelia",
    "rich",
    "happy",
]
embs = (W1.T + W2) / 2.0
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
print(X.shape, idx)

result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

        