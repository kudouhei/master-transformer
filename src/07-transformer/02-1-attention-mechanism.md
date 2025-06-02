# Attention mechanism

## Attention more formally

![Attention mechanism](./images/attention-1.png)

The Fig. above illustrates this flow of information in an entire causal self-attention layer, in which this same attention computation happens in parallel at each token position $i$. Thus a self-attention layer maps input sequences $(x_1, ..., x_n)$ to output sequences of the same length $(a_1, ..., a_n)$.

### Simplified version of attention
At its heart, attention is really just a weighted sum of context vectors, with a lot of complications added to how the weights are computed and what gets summed.

Let's describe a simplified version of attention, in which the attention output $a_i$ at token position $i$ is simply the weighted sum of all the representations $x_j$, for all $j \leq i$; we’ll use $\alpha_{ij}$ to mean how much $x_j$ should contribute to $a_i$:

**Simplified version: $a_i = \sum \alpha_{ij} x_j$**


We compute similarity scores via dot product. We’ll normalize these scores with a softmax to create the vector of weights $\alpha_{ij}$, for all $j \leq i$.

**Simplified Version: $score(x_i ,x_j ) = x_i ·x_j$**
**Normalized: $\alpha_{ij} = softmax(score(x_i ,x_j ))$**

Thus in the Fig we compute $a_3$ by computing three scores: $x_3 ·x_1$, $x_3 ·x_2$, and $x_3 ·x_3$, normalizing them by a softmax, and using the resulting probabilities as weights indicating each of their proportional relevance to the current position $i$.

### A single attention head using query, key, and value matrices
Now that we’ve seen a simple intuition of attention, let’s introduce the actual attention head, the version of attention that’s used in transformers.

The attention head allows us to distinctly represent three different roles that each input embedding plays during the course of the attention process:

- **Query**: As the current element being compared to the preceding inputs.
- **Key**: In its role as a preceding input that is being compared to the current element to determine a similarity weight.
- **Value**: As a value of a preceding element that gets weighted and summed up to compute the output for the current element.

To capture these three different roles, transformers introduce weight matrices $W^Q$, $W^K$, and $W^V$. These weights will project each input vector $x_i$ into a representation of its role as a key, query, or value:

**$q_i = W^Q x_i$**
**$k_i = W^K x_i$**
**$v_i = W^V x_i$**

- When we are computing the similarity of the current element $x_i$ with some prior element $x_j$, we’ll use the **dot product** between the current element’s query vector $q_i$ and the preceding element’s key vector $k_j$. 
- Furthermore, the result of a dot product can be an arbitrarily large (positive or negative) value, and exponentiating large values can lead to numerical issues and loss of gradients during training.
- To avoid this, we **scale the dot product by a factor related to the size of the embeddings, via dividing by the square root of the dimensionality of the query and key vectors ($d_k$).**
- The ensuing softmax calculation resulting in $\alpha_{ij}$ remains the same, but the output calculation for head $i$ is now based on a weighted sum over the value vectors $v$.

Here’s a final set of equations for computing self-attention for a single self-attention output vector $a_i$ from a single input vector $x_i$. This version of attention computes $a_i$ by summing the values of the prior elements, each weighted by the similarity of its key to the query from the current element:

$$
q_i = W^Q x_i; k_j = W^K x_j; v_j = W^V x_j
$$

$$
score(x_i ,x_j ) = \frac{x_i ·x_j}{\sqrt{d_k}} = \frac{q_i ·k_j}{\sqrt{d_k}}
$$

$$
\alpha_{ij} = softmax(score(x_i ,x_j ))
$$

$$
head_i = \sum_{j \leq i} \alpha_{ij} v_j
$$

$$
a_i = head_i W^O
$$

![Attention mechanism](./images/attention-2.png)

**Note that we’ve also introduced one more matrix, $W^O$, which is right-multiplied by the attention head.**  This is necessary to reshape the output of the head. 
- The input to attention $x_i$ and the output from attention $a_i$ both have the same dimensionality [1 × d]. We often call d the **model dimensionality**.

Let’s talk **shapes**. 

How do we get from $[1 × d]$ at the input to $[1 × d]$ at the output? Let’s look at all the internal shapes.
               
1. We’ll have a dimension $d_k$ for the key and query vectors.
2. The query vector and the key vector are both dimensionality **$1 × d_k$**, so we can take their dot product **$q_i · k_j$** to produce a scalar.
3. We’ll have a separate dimension $d_v$ for the value vectors.
4. **The transform matrix $W^Q$ has shape **$[d × d_k]$**, $W^K$ is **$[d × d_k]$**, and $W^V$ is **$[d × d_v]$**.**
5. So the output of $head_i$ in equation is of shape **$[1 × d_v]$**.
6. To get the **desired output shape $[1 × d]$** we’ll need to reshape the head output, and so **$W^O$ is of shape $[d_v × d]$**.


### Multi-head attention
The intuition is that each head might be attending to the context for different purposes: heads might be specialized to represent different linguistic relationships between context elements and the current token, or to look for particular kinds of patterns in the context.

So in **multi-head attention** we have $A$ separate attention heads that reside in parallel layers at the same depth in a model, each with its own set of parameters that allows the head to model different aspects of the relationships among inputs.

Thus, each head $i$ in a self-attention layer has its own set of key, query and value matrices: $W^{K_i}$, $W^{Q_i}$ and $W^{V_i}$. These are used to project the inputs into separate key, value, and query embeddings for each head.

When using multiple heads the model dimension d is still used for the input and output, the key and query embeddings have dimensionality $d_k$, and the value embeddings are of dimensionality $d_v$ (again, in the original transformer paper $d_k = d_v= 64$, $A = 8$, and $d = 512$).

**Thus for each head $i$, we have weight layers $W^{Q_i}$ of shape $[d × d_k]$, $W^{K_i}$ of shape $[d × d_k]$, and $W^{V_i}$ of shape $[d × d_v]$.**

Below are the equations for attention augmented with multiple heads:

$$
q^c_i = W^{Q_c} x_i; k^c_j = W^{K_c} x_j; v^c_j = W^{V_c} x_j
$$

$$
score^c(x_i ,x_j ) = \frac{q^c_i ·k^c_j}{\sqrt{d_k}}
$$

$$
\alpha^c_{ij} = softmax(score^c(x_i ,x_j ))
$$

$$
head^c_i = \sum_{j \leq i} \alpha^c_{ij} v^c_j
$$

$$
a_i= (head^1 ⊕ head^2... ⊕ head^A)W^O
$$

$$ 
Multi-head attention: a_i= (x_i, [x_1, ..., x_N])
$$

- The output of each of the A heads is of shape $1 × d_v$ , and so the output of the multi-head layer with A heads consists of A vectors of shape $1 × d_v$. 
- These are concatenated to produce a single output with dimensionality $1 × hd_v$.
- Then we use yet another linear projection $W^O$ ∈ $R^{Ad_v ×d}$ to reshape it, resulting in the multi-head attention vector $a_i$ with the correct output shape $[1×d]$ at each input $i$.


![Multi-head attention](./images/attention-3.png)


### Attention mechanism workflow

![Attention mechanism](./images/attention.png)

The figure depicts different attention mechanisms, starting with a simplified version of self-attention before adding the trainable weights. The causal attention mechanism adds a mask to self-attention that allows the LLM to generate one word at a time. Finally, multi-head attention organizes the attention mechanism into multiple heads, allowing the model to capture various aspects of the input data in parallel.

**The problem with modeling long sequences**
we can’t simply translate a text word by word due to the grammatical structures in the source and target language.

![Long sequence problem](./images/translate.png)

To address this problem, it is common to use a deep neural network with two submodules, an encoder and a decoder. The job of the encoder is to first read in and process the entire text, and the decoder then produces the translated text.

Before the advent of transformers, recurrent neural networks (RNNs) were the most popular encoder–decoder architecture for language translation. An RNN is a type of neural network where outputs from previous steps are fed as inputs to the current step.

The encoder takes a sequence of tokens from the source language as input, where a hidden state (an intermediate neural network layer) of the encoder encodes a compressed representation of the entire input sequence. Then, the decoder uses its current hidden state to begin the translation, token by token.

### Capture data dependencies with attention mechanisms

![Attention mechanism](./images/bahdanau.png)

Using an attention mechanism, the text-generating decoder part of the network can access all input tokens selectively. This means that some input tokens are more important than others for generating a given output token. The importance is determined by the attention weights, which we will compute later. 

**Self-attention** is a mechanism that allows each position in the input sequence to consider the relevancy of, or “attend to,” all other positions in the same sequence when computing the representation of a sequence. Self-attention is a key component of contemporary LLMs based on the transformer architecture, such as the GPT series.

In self-attention, the goal is to calculate context vectors `z(i)` for each element `x(i)`  in the input sequence. A context vector can be interpreted as an enriched embedding vector. Context vectors play a crucial role in self-attention. Their purpose is to create enriched representations of each element in an input sequence.

### Self-attention with trainable weights
![Self-attention](./images/coding-attention/without-weights10.png)

In self-attention, we transform the input vectors in the input matrix $X$ with the three weight matrices, $W_q$ , $W_k$ , and $W_v$ . The new compute the attention weight matrix based on the resulting queries (Q) and keys (K). Using the attention weights and values (V), we then compute the context vectors (Z). 

```
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa = SelfAttention(d_in, d_out)
print(sa(inputs))
```

### Causal attention
Causal attention, also known as masked attention, is a specialized form of self-attention. It restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores.
In contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once.

![Causal attention](./images/coding-attention/mask01.png)

In causal attention, we mask out the attention weights above the diagonal such that for a given input, the LLM can’t access future tokens when computing the context vectors using the attention weights. For example, for the word “journey” in the second row, we only keep the attention weights for the words before (“Your”) and in the current position (“journey”).

**Apply a causal mask to the attention**
![Causal attention](./images/coding-attention/mask02.png)

- However, if the **mask were applied after softmax**, like above, it would disrupt the probability distribution created by softmax
- Softmax ensures that all output values sum to 1
- Masking after softmax would require re-normalizing the outputs to sum to 1 again, which complicates the process and might lead to unintended effects


Instead of zeroing out attention weights above the diagonal and renormalizing the results, we can mask the unnormalized attention scores above the diagonal with negative infinity before they enter the softmax function:

![Causal attention](./images/coding-attention/mask03.png)

**Masking additional attention weights with dropout**
Dropout in deep learning is a technique where randomly selected hidden layer units are ignored during training, effectively “dropping” them out. This method helps prevent overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units.

**Dropout** in the attention mechanism is typically applied at two specific times: **after calculating the attention weights** or **after applying the attention weights to the value vectors**.

![Dropout](./images/coding-attention/mask04.png)

- Furthermore, in this specific example, we use a dropout rate of 50%, which means randomly masking out half of the attention weights. 
- When we train the GPT model later, we will use a lower dropout rate, such as 0.1 or 0.2
- If we apply a dropout rate of 0.5 (50%), the non-dropped values will be scaled accordingly by a factor of 1/0.5 = 2
- The scaling is calculated by the formula 1 / (1 - dropout_rate)

**Implementing a compact causal attention class**

```
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout) # New, add a dropout layer

        # 注册一个持久的注意力掩码（mask）“生成一个上三角矩阵作为注意力掩码，并注册为模型的固定属性，确保自回归模型不会作弊看到未来信息”
        # New, The register_buffer call is also a new addition, more information is provided in the following text).
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape # new batch dimension b; batch size（批大小）;token数量; embedding维度

        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)

        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
```

### Multi-head attention
The term “multi-head” refers to dividing the attention mechanism into multiple “heads,” each operating independently. In this context, a single causal attention module can be considered single-head attention, where there is only one set of attention weights processing the input sequentially.

**Stacking multiple single-head attention layers**
Implementing multi-head attention involves creating multiple instances of the self-attention mechanism, each with its own set of trainable weights.

![Multi-head attention](./images/coding-attention/multi01.png)

For example in the figure above. The multi-head attention module includes two single-head attention modules stacked on top of each other. So, instead of using a single matrix $W_v$ for computing the value matrices, in a multi-head attention module with two heads, we now have two value weight matrices: $W_{v1}$ and $W_{v2}$. The same applies to the other weight matrices, $W_Q$ and $W_k$. We obtain two sets of context vectors $Z_1$ and $Z_2$ that we can combine into a single context vector matrix $Z$.

**Implementing a multi-head attention class**
We simply stack multiple single-head attention modules to obtain a multi-head attention module:

```
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

```

![Multi-head attention](./images/coding-attention/multi02.png)

Using the MultiHeadAttentionWrapper, we specified the number of attention heads (num_heads). If we set num_heads=2, as in this example, we obtain a tensor with two sets of context vector matrices. In each context vector matrix, the rows represent the context vectors corresponding to the tokens, and the columns correspond to the embedding dimension specified via d_out=4. We concatenate these context vector matrices along the column dimension. Since we have two attention heads and an embedding dimension of 2, the final embedding dimension is 2 × 2 = 4.

**Implementing multi-head attention with weight splits**

```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
```

