# Singular Value Decomposition (SVD)

!!! info "Where this shows up"
    PCA, low-rank matrix approximation, pseudo-inverses / least squares,
    recommender systems (matrix factorization), and as the numerical
    backbone under a lot of "compress this big matrix" tricks in ML papers.

## Definition

Any real matrix $A \in \mathbb{R}^{m \times n}$ can be factored as

$$
A = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are
  orthogonal ($U^\top U = I$, $V^\top V = I$).
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal (in the rectangular
  sense — zero off the main diagonal) with non-negative entries
  $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r \geq 0$, the **singular
  values**, where $r = \mathrm{rank}(A)$.

Columns of $U$ are the **left singular vectors**, columns of $V$ the
**right singular vectors**.

## Where U, V, Σ come from

- The columns of $V$ are eigenvectors of $A^\top A$; the columns of $U$
  are eigenvectors of $A A^\top$.
- The singular values are the square roots of the (shared, non-negative)
  eigenvalues of $A^\top A$ and $A A^\top$: $\sigma_i = \sqrt{\lambda_i}$.
- So SVD generalizes eigendecomposition to non-square, non-symmetric
  matrices — every matrix has an SVD, but not every matrix has an
  eigendecomposition.

## Geometric picture

$A$ maps the unit sphere in $\mathbb{R}^n$ to an ellipsoid in
$\mathbb{R}^m$: $V^\top$ rotates the input, $\Sigma$ stretches each axis
by $\sigma_i$, and $U$ rotates the result into place. The $\sigma_i$ are
literally the lengths of the ellipsoid's semi-axes.

## Low-rank approximation (Eckart–Young)

Writing $A = \sum_{i=1}^r \sigma_i u_i v_i^\top$ (sum of rank-1 terms,
largest $\sigma_i$ first), the best rank-$k$ approximation of $A$ in both
Frobenius and spectral norm is just truncating the sum:

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top
$$

This is the "why" behind SVD-based compression: keep the top-$k$ singular
triples, drop the rest, and you have the closest possible rank-$k$
matrix — with error $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$.

## Relation to PCA

For a centered data matrix $X$ (rows = samples, columns = features), PCA's
principal components are exactly the right singular vectors $V$ of $X$,
and the projected scores are $U \Sigma$. In practice, SVD on $X$ directly
is more numerically stable than eigendecomposing the covariance matrix
$X^\top X$.

## Pseudo-inverse

The Moore-Penrose pseudo-inverse falls out directly:

$$
A^+ = V \Sigma^+ U^\top
$$

where $\Sigma^+$ inverts the non-zero singular values in place and
transposes the shape. This gives the least-squares solution to
$Ax = b$ even when $A$ isn't square or invertible.

```python
import numpy as np

A = np.random.randn(50, 20)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

k = 5
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]  # best rank-k approximation
```
