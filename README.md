# causal-dragonnet-advanced

This is an **advanced** dummy project that mirrors a DragonNet-style continuous-treatment setup similar to what you described:

- Shared representation: `z = g(x)` from numeric + categorical embeddings  
- GPS head: Gaussian `mu(x), sigma(x)` modeling `T | X`  
- Outcome head: `logits = q(z, t)` with:
  - `t_mlp(t)` (treatment embedding)
  - optional `gamma(z) * t` term (toggle)
  - optional `cat_offset(x)` term (toggle)
- Loss: `BCEWithLogits` + GPS Gaussian NLL + targeted regularizer (`epsilon`) + monotonicity finite-difference penalty

## Colab: you do NOT need a notebook
Colab can run python scripts directly:

```python
!git clone <YOUR_REPO_URL>
%cd causal-dragonnet-advanced
!pip install -r requirements.txt
!python -m src.train
```

## Notebook included
A `colab_demo.ipynb` is included for convenience.
