# gobots
The GoBots to Hugging Face's Transformers

##### Table of Contents
-[Architectural Components](#architectural-components)
  - [positional encodings](#positional-encodings)
    - [absolute positional encodings](#absolute-positional-encodings)
    - [T5 relative positional encodings](#t5-relative-positional-encodings)
    - [ALiBi](#alibi)

## Architectural Components

In what follows, unless otherwise stated, we make make the assumption that inputs to the model take the form `[batchsize, sequence_len, embedding_dimension]`
with generic shape `[n, m, d]`.

### Positional Encodings

much of the insights provided bellow can be found [here](https://arxiv.org/abs/2305.19466).

#### absolute positional encodings

absolute positional encoding add a fixed vector to each element of the input sequence. A popular
version is the sinusoidal encoding (introduced int he original transformer [paper](https://arxiv.org/abs/1706.03762)) which is applied to the inputs before being fed through the transformer blocks

$$
PE_{(pos,i)} = \begin{cases}
                  \sin\left( \frac{m}{10000^{i/d}} \right) & i \text{ is even} \\
                  \cos\left( \frac{m}{10000^{(i - 1)/d}} \right) & i \text{ is odd}
\end{cases}
$$

where $x_m^{(i)}$ is the ith component of the mth input token embedding.

pros:
- easy to implement.
- relatively fast, it is just one addition.

cons:
- does not seem to generalize well to unseen input sequence lenghts
- does not encourage attention to any particular positions in the sequence.
- is not explicitly included in the attention mechanism.

#### Rotary Positional encodings

Rotary positional encodings (RoPE), introduced in the RoFormer [paper](https://arxiv.org/abs/2104.09864) work by rotating the query and
key vectors in the attention mechanism by angles proportional to the position in the sequence. The rotation matrix can be parameterized as

$$
\mathbf{R}^d_{\Theta, m} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \dots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \dots & 0 & 0 \\
0 & 0 &\cos m\theta_2 & -\sin m\theta_2 & \dots & 0 & 0 \\
0 & 0 &\sin m\theta_2 & \cos m\theta_2 & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & 0 & 0 \\
0 & 0 & 0 & 0 & \dots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \dots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \\
\end{pmatrix}
$$

where $\theta_i = 10000^{-2(i -1)/d}$ for $i = 1,2,3,\dots,d/2$.t

pros:
- widely adopted
- incorporates positional information directly into the attention mechanism
- does not introduce any parameters into the model
- combines some benefits of relative and absolute encodings

cons:
- more expensive then some encodings (multiplication slower than addition)
- does not scale as well to unseen sequence lengths

#### T5 relative positional encodings

These positional encodings were introduced in the [T5 paper](https://arxiv.org/abs/1910.10683). Here a learned bias is added to the attention scores.
The parametrs are shared across layers and attention heads. A fixed number of relative distance buckets are used (32 in the original implementation) up to a maximum distance where the buckets scale logarithmically with distance.

pros:
- incorporated in the attention mechanism
- each attention head gets its own set of embeddings which are shared across layers, making this a fast and parameter efficient encoding mechanism
- generalizes well to unseen sequence lengths
- biases attention has two peaks in relative distance for near and distant tokens.

cons:
- slower then NoPE, ALiBi, and RoPE

#### ALiBi

The term ALiBi comes from the phrase "attention with linear biases" and was introduced [here](https://arxiv.org/pdf/2108.12409). As the name suggests
this method encodes relative positions in the attention mechanism by subtracting a fixed bias term to the attention score which scales linearly with distance.
These biases are not learned and each attention head is given its own bias (shared by all layers).

pros:
- Parameter efficient, no learned parameters
- incorporated into the attention mechanism
- fast to compute

cons:
- biases attention toward nearby tokens.
