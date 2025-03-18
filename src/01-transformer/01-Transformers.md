# Overview of the Transformer architecture

The attention mechanism is an important part of these models and plays a very crucial role. Before Transformer models, the attention mechanism was proposed as a helper for improving conventional DL models such as RNNs. To understand Transformers and their impact on NLP, we will first study the attention mechanism.

## Attention mechanism

The attention mechanism is a mechanism that allows the model to focus on the most important parts of the input sequence. It is a mechanism that allows the model to attend to the most important parts of the input sequence.

### Self-attention

Self-attention is a mechanism that allows the model to attend to the most important parts of the input sequence. It is a mechanism that allows the model to attend to the most important parts of the input sequence.

Self-attention is a basic form of a scaled self-attention mechanism. This mechanism uses an input matrix and produces an attention score between various items Q is known as the query, K is known as the key, and V is noted as the value.

![Self-attention](../images/self-attention.png)
So, X is the input word sequence, and we calculate three values from that which is **Q(Query)**, **K(Key)** and **V(Value)**.

- **Query (Q)**: What we're looking for
- **Key (K)**: What we match against
- **Value (V)**: The information to extract

The task is to find the important words from the Keys for the Query word. This is done by passing the query and key to a mathematical function (usually matrix multiplication followed by softmax). The resulting context vector for Q is the multiplication of the probability vector obtained by the softmax with the Value.

When the Query, Key, and Value are all generated from the same input sequence **X**, it is called **Self-Attention**.

### Scaled dot-product attention mechanism
A scaled dot-product attention mechanism is very similar to a self-attention (dot-product) mechanism except it uses a scaling factor.

### Multi-head attention

Multi-head attention is a mechanism that allows the model to attend to the most important parts of the input sequence. It is a mechanism that allows the model to attend to the most important parts of the input sequence.

![Multi-head Attention](../images/multi-head.png)

Mathematical Formulation: For each head ‘h’, the attention mechanism is defined as:

![Mathematical Formulation](../images/multi-head-formulation.png)
where `Q`, `K`, and `V` are the query, key, and value matrices, respectively, and `d_k` is the dimension of the key vectors.

In multi-head attention, these heads are computed in parallel:

![Mathematical Formulation](../images/multi-head-formulation2.png)
where each head `head_i` is an attention function, and `W^o` is a learned linear transformation.


### A Transformer architecture

A Transformer architecture is a type of architecture that is used to build NLP models. It is a type of architecture that is used to build NLP models.

![Transformer Architecture](../images/transformer.png)

#### Tokens and embeddings

Tokens are the basic units of the input sequence. They are the individual words or characters in the input sequence.

Embeddings are the vectors that represent the tokens. They are the vectors that represent the tokens.

![Tokens and embeddings](../images/tokens.png)
Language models deal with text in small chunks called tokens. For the language model to compute language, it needs to turn tokens into numeric representations called embeddings.

**Token embeddings**

![Token embeddings](../images/embeddings.png)
A language model holds an embedding vector associated with each token in its tokenizer.

#### An Overview of Transformer Models

![An Overview of Transformer Models](../images/llm-inside.png)
A Transformer LLM is made up of a tokenizer, a stack of Transformer blocks, and a language modeling head.

**Parallel Token Processing and Context Size**

![Parallel Token Processing and Context Size](../images/parallel.png)
When generating text, it’s important to cache the computation results of previous tokens instead of repeating the same calculation over and over again.

![Parallel Token Processing and Context Size](../images/parallel1.png)
The bulk of the Transformer LLM processing happens inside a series of Transformer blocks, each handing the result of its processing as input to the subsequent block.

A **Transformer block** is made up of two successive components:
1. The **attention layer** is mainly concerned with incorporating relevant information from other input tokens and positions
2. The **feedforward layer** houses the majority of the model’s processing capacity

![Transformer Block](../images/transformer-block.png)









