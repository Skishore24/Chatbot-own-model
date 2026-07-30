# 🤖 GENKIT AI — PyTorch AI Pipeline & Mathematics

## Mathematical Formulations

### 1. Grouped-Query Attention (GQA)
$$H_Q = 12, \quad H_{KV} = 4, \quad G = 3$$
$$\text{GQA}(Q, K, V) = \text{Concat}\Big(\text{head}_1, \dots, \text{head}_{H_Q}\Big) W^O$$
Reduces KV-cache memory usage by 75% while maintaining attention accuracy.

### 2. Rotary Position Embedding (RoPE)
$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \quad \theta_i = (10000 \cdot S)^{-2(i-1)/d}$$

### 3. SwiGLU Activation
$$\text{SwiGLU}(x) = \left(\text{Swish}(x W_g) \odot x W_1\right) W_2$$

### 4. RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 5. Reciprocal Rank Fusion (RRF)
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
