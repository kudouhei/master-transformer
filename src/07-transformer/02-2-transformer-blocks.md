# Transformer blocks

The self-attention calculation lies at the core of what’s called a transformer block, which, in addition to the self-attention layer, includes three other kinds of layers: 
1. a feedforward layer, 
2. residual connections, 
3. normalizing layers (colloquially called “layer norm”).

![Transformer block](./images/block-1.png)

### Residual stream

In the residual stream viewpoint, we consider the processing of an individual token $i$ through the transformer block as a single stream of $d$-dimensional representations for token position $i$. This residual stream starts with the original input vector, and the various components read their input from the residual stream and add their output back into the stream.

This initial embedding gets passed up (by **residual connections**), and is progressively added to by the other components of the transformer: the **attention layer** that we have seen, and the **feedforward layer** that we will introduce.

**Feedforward layer**
The feedforward layer is a fully-connected 2-layer network, i.e., one hidden layer, two weight matrices. The weights are the same for each token position $i$ , but are different from layer to layer. It is common to make the dimensionality $d_{ff}$ of the hidden layer of the feedforward network be larger than the model dimensionality $d$. (For example in the original transformer model, $d = 512$ and $d_{ff} = 2048$.)

$$
\text{FFN}(x_i) = ReLU(x_iW_1 + b_1)W_2 + b_2
$$

### Layer Norm






