## FAQ

#### 1. In the Transformer attention mechanism, why do we divide by $\sqrt{d_k}$ in the scaled dot-product attention?

Mathematical reason: If queries and keys are independent with mean 0 and variance 1, their dot product has mean 0 and variance = ${d_k}$.

Practical reason: Large dot products push softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ helps to keep the dot product in a reasonable range.

We divide by $\sqrt{d_k}$ to scale the dot products. Since the variance of the dot product grows linearly with dimension $d_k$, dividing by $\sqrt{d_k}$ (the standard deviation) keeps the variance constant at 1. This prevents the softmax from entering saturation regions where gradients vanish, ensuring stable training.

#### 2. Why do we need multi-head attention instead of just one attention head?

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

Key points:
- Single head limitation: One attention head averages out different types of relationships, potentially losing important information
- Multiple perspectives: Different heads can learn to focus on different aspects (e.g., head1 captures subject-verb relations, head2 captures adjective-noun relations, head3 captures long-distance dependencies)
- Parallel computation: All heads compute simultaneously, then concatenate results

Like having multiple experts examine a problem from different angles instead of relying on a single generalist.

#### 3. What is the difference between self-attention and cross-attention?
Self-attention computes relationships within the same sequence, while cross-attention computes relationships between two different sequences.

Transformer architecture mapping:
- Encoder: Self-attention only
- Decoder: Masked self-attention + Cross-attention (with encoder outputs)

#### 4. Why do transformers use residual connections and layer normalization?
Residual connections and layer normalization work together to enable stable training of very deep networks by preventing gradient vanishing/exploding and maintaining consistent activation scales.

**Residual connections (Add):**
$$Output = Layer(x) + x$$

- Preserve information: Original input information isn't lost through transformations.

**Layer normalization (LayerNorm):**
$$Output = \gamma \frac{x - \mu}{\sigma} + \beta$$

- Stabilizes training: Keeps activations with consistent mean and variance across layers
- Faster convergence: Reduces sensitivity to parameter initialization and learning rate