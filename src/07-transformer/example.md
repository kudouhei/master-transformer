## Example

> Complete Pipeline: "the cat sat on the mat"

#### Step 1: Build Vocabulary

From training data, collect unique words + special tokens:

| Index | Token  |
|-------|--------|
| 0     | [PAD]  |
| 1     | [UNK]  |
| 2     | [CLS]  |
| 3     | [SEP]  |
| 4     | the    |
| 5     | cat    |
| 6     | sat    |
| 7     | on     |
| 8     | mat    |

vocab_size = 9

#### Step 2: Tokenization → Index Sequence

```
"the cat sat on the mat" → [4, 5, 6, 7, 4, 8]
```

#### Step 3: Embedding + Positional Encoding

- Embedding matrix E: [vocab_size, d_model] (d_model=4), for example:

| Index | Token | Vector        |
|-------|-------|---------------|
| 4     | the   | [1, 0, 1, 0]  |
| 5     | cat   | [0, 1, 0, 1]  |
| 6     | sat   | [1, 1, 0, 0]  |
| 7     | on    | [0, 0, 1, 1]  |
| 8     | mat   | [1, 0, 0, 1]  |

- Look up each index → token vectors

- Add positional encoding (simplified: pos i = [i,i,i,i] normalized)

| Pos | Encoding        |
|-----|----------------|
|  0  | [0, 0, 0, 0]   |
|  1  | [1, 1, 1, 1]   |
|  2  | [2, 2, 2, 2]   |
|  3  | [3, 3, 3, 3]   |
|  4  | [4, 4, 4, 4]   |
|  5  | [5, 5, 5, 5]   |

- Final Input X = Embedding + Positional Encoding (scaled):

| Pos | Token | 简化后(方便示意)Final Input x (4-dim)   |
|-----|-------|-------------------------|
|  0  | the   | [2, 0, 1, 0]            |
|  1  | cat   | [0, 2, 0, 1]            |
|  2  | sat   | [1, 1, 1, 0]            |
|  3  | on    | [0, 0, 2, 2]            |
|  4  | the   | [2, 0, 1, 0]            |
|  5  | mat   | [1, 1, 0, 2]            |


#### Step 4: Transformer Layer (Single Head, d_k=3)

**4.1 Generate Q, K, V matrices**
Assume trained weight matrices (simplified):

| Pos | q (3-dim) | k (3-dim) | v (3-dim) |
|-----|-----------|-----------|-----------|
| 0   | [2,0,1]   | [2,1,0]   | [0,2,1]   |
| 1   | [0,2,0]   | [0,0,2]   | [2,0,0]   |
| 2   | [1,1,1]   | [1,1,1]   | [1,1,1]   |
| 3   | [0,0,2]   | [0,2,0]   | [0,0,2]   |
| 4   | [2,0,1]   | [2,1,0]   | [0,2,1]   |
| 5   | [1,1,0]   | [1,0,1]   | [1,1,0]   |

**4.2 Attention for position 4 (predicting next word)**
`q_4 = [2,0,1]`

**Scores (causal: j ≤ 4):**

- `j=0`: k₀=[2,1,0] → dot(2x2+0x1+1x0)=4 → 4/√3=2.31

- `j=1`: k₁=[0,0,2] → dot=2 → 1.15

- `j=2`: k₂=[1,1,1] → dot=3 → 1.73

- `j=3`: k₃=[0,2,0] → dot=0 → 0

- `j=4`: k₄=[2,1,0] → dot=4 → 2.31

**Softmax:**
`Softmax = exp(scores) / sum(exp(scores))`

α = [0.30, 0.10, 0.20, 0.05, 0.35] (sum=1)

**4.3 Weighted Sum**

`head₄ = Σ αⱼ × vⱼ`
= `0.30×[0,2,1] + 0.10×[2,0,0] + 0.20×[1,1,1] + 0.05×[0,0,2] + 0.35×[0,2,1]`
= `[0.4, 1.5, 0.95]`

**4.4 Output Projection (W^O: 3→4)**

Assume W^O trained:

W^O = [
1, 0, 1, 0; 
0, 1, 0, 1; 
1, 0, 0, 1]

`a₄ = head₄ × W^O = [0.4, 1.5, 0.95] × [1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 0, 1]` = `[0.8, 1.2, 0.9, 1.5]`

**4.5 Residual Connection**

`output₄ = a₄ + x₄ = [0.8,1.2,0.9,1.5] + [2,0,1,0] = [2.8, 1.2, 1.9, 1.5]`


**4.6 Layer Normalization (simplified)**

`μ = (2.8+1.2+1.9+1.5)/4 = 1.85`

`σ = sqrt(((2.8-1.85)^2+(1.2-1.85)^2+(1.9-1.85)^2+(1.5-1.85)^2)/4) = 0.55 ≈ 0.6`

`norm₄ = (output₄ - μ)/σ ≈ [1.58, -1.08, 0.08, -0.58]`

**4.7 Feed-Forward Network**

Assume W₁(4x8), W₂(8x4), ReLU:

`hidden₄ = ReLU(norm₄ × W₁ + b₁)` -> `[1.2, 0, 2.3, 0.5, 0, 1.1, 0, 0.8] (8-dim)`

`ffn₄ = hidden₄ × W₂ + b₂ → [1.5, 2.1, 1.8, 2.3]`

**4.8 Second Residual**

`final₄ = ffn₄ + norm₄ = [1.5,2.1,1.8,2.3] + [1.58,-1.08,0.08,-0.58] = [3.08, 1.02, 1.88, 1.72]`


#### Step 5: Language Model Head


LM Head = a linear layer that transforms a vector of size d_model into a vector of size vocab_size. (LM Head = 一个线性层，把 d_model 维的向量变成 vocab_size 维的向量 )

`logits = final_hidden × W_lm + b_lm`

Assume W_lm [4×9] ([d_model, vocab_size]) and b_lm [9] ([vocab_size]) trained:

`logits = final₄ × W_lm + b_lm` = `[1.5, 0.8, -0.5, -1.2, 2.1, 0.3, -0.8, 1.2, 3.5]`

**Softmax**:

`e^1.5=4.48, e^0.8=2.23, e^-0.5=0.61, e^-1.2=0.30, e^2.1=8.17, e^0.3=1.35, e^-0.8=0.45, e^1.2=3.32, e^3.5=33.12`

`sum = 4.48+2.23+0.61+0.30+8.17+1.35+0.45+3.32+33.12 = 54.03`

**Probabilities**: `softmax(logits) = e^logits / Σ e^logits`

- [PAD]: 4.48/54.03 = 0.083

- [UNK]: 2.23/54.03 = 0.041

- [CLS]: 0.61/54.03 = 0.011

- [SEP]: 0.30/54.03 = 0.006

- the: 8.17/54.03 = 0.151

- cat: 1.35/54.03 = 0.025

- sat: 0.45/54.03 = 0.008

- on: 3.32/54.03 = 0.061

- **mat: 33.12/54.03 = 0.613**

#### Cross-Entropy Loss

if the true next word is "mat", but the model predicted "mat" with higher probability, then the cross-entropy loss is:

`loss = -log(probs[8]) = -log(0.613) = 0.47`

The loss is adjusted through backpropagation: adjust:

- W_lm and b_lm

- all parameters of previous Transformer layers

- even the embedding matrix E

to make the probability of "mat" higher next time

#### Step 6: Output

Predicted next word: "mat" with **61.3%** confidence.