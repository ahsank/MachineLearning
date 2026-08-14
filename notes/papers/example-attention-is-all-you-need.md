# Attention Is All You Need (Vaswani et al., 2017)

!!! info "Reference"
    Vaswani, A. et al. *Attention Is All You Need*. NeurIPS 2017.
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

This is a template/example note — copy this file to start a new paper
summary and delete the parts that don't apply.

## TL;DR

Replaces recurrence and convolution in sequence transduction with
self-attention alone. Faster to train (more parallelizable) and set a new
state of the art on translation at the time.

## Problem

RNN/LSTM encoder-decoder models process tokens sequentially, which limits
parallelism and makes it hard to relate distant tokens (long dependency
chains). Attention mechanisms had already been used *on top of* recurrence
to bridge distant positions — this paper asks whether recurrence is needed
at all.

## Key idea: scaled dot-product attention

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q$, $K$, $V$ are learned linear projections of the input.
- The $\sqrt{d_k}$ scaling keeps dot products from growing too large in
  magnitude, which would push softmax into regions with tiny gradients.

**Multi-head attention** runs $h$ of these in parallel with different
learned projections, then concatenates the results — letting the model
attend to information from different representation subspaces at once.

## Architecture

- Encoder: stack of self-attention + feed-forward blocks, each wrapped in a
  residual connection and layer norm.
- Decoder: same, plus a masked self-attention (so a position can't attend
  to future tokens) and cross-attention over the encoder output.
- No recurrence or convolution, so positional information is injected
  explicitly via **positional encodings** added to the input embeddings.

```python
# Sketch of scaled dot-product attention
def attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
```

## Results

State-of-the-art BLEU on WMT 2014 English-to-German and English-to-French,
trained in a fraction of the time reported for prior best models thanks to
parallelization across positions.

!!! note "Why it mattered later"
    This architecture became the base block for BERT, GPT, and most
    subsequent large language models — the encoder or decoder stack used
    almost unmodified, scaled up.

## Open questions / things to dig into

- How does the $O(n^2)$ attention cost in sequence length get addressed in
  follow-up work? (sparse/linear attention, FlashAttention, etc.)
- Why do learned positional encodings vs. the sinusoidal ones used here
  matter in practice?
