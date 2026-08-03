# Diffusion Models From Scratch

A from-scratch implementation of core diffusion model formulations, built up progressively from denoising diffusion probabilistic models to latent diffusion. Each subdirectory implements one formulation as a self-contained module, following the corresponding paper's formal definitions.

---

## Repository Structure

| Directory | Paper | Description |
|---|---|---|
| [`DDPM/`](./DDPM) | Ho et al., 2020 | Denoising Diffusion Probabilistic Models — the base forward/reverse diffusion process |
| [`DDIM/`](./DDIM) | Song et al., 2020 | Denoising Diffusion Implicit Models — non-Markovian, deterministic fast sampling |
| [`Classifier-Free Guidance/`](./Classifier-Free%20Guidance) | Ho & Salimans, 2021 | Conditional generation without an auxiliary classifier |
| [`Latent Diffusion Model/`](./Latent%20Diffusion%20Model) | Rombach et al., 2022 | Diffusion in a compressed latent space (Stable Diffusion's foundation) |

---

## Model Descriptions

### 1. DDPM — Denoising Diffusion Probabilistic Models

**Forward process.** A Markov chain gradually adds Gaussian noise to data `x0` over `T` steps according to a variance schedule `β1, ..., βT`:

```
q(xt | xt-1) = N(xt; √(1-βt)·xt-1, βt·I)
```

This admits a closed form for sampling `xt` directly from `x0`, using `αt = 1-βt` and `ᾱt = Π αs`:

```
q(xt | x0) = N(xt; √ᾱt · x0, (1-ᾱt)·I)
xt = √ᾱt · x0 + √(1-ᾱt) · ε,   ε ~ N(0, I)
```

**Reverse process.** A neural network `εθ` is trained to predict the noise component at each step, and the reverse process is parameterized as:

```
pθ(xt-1 | xt) = N(xt-1; μθ(xt, t), Σθ(xt, t))
```

**Training objective.** The simplified loss reduces to predicting the injected noise:

```
L_simple = E[t, x0, ε] [ || ε - εθ(xt, t) ||² ]
```

**Sampling.** Iteratively denoise from pure Gaussian noise `xT ~ N(0, I)` down to `x0` over `T` steps.

---

### 2. DDIM — Denoising Diffusion Implicit Models

DDIM defines a non-Markovian forward process that shares the same marginals `q(xt | x0)` as DDPM, but allows a **deterministic**, generalized reverse process:

```
x(t-1) = √ᾱ(t-1) · x̂0 + √(1-ᾱ(t-1) - σt²) · εθ(xt, t) + σt·ε
```

where the predicted clean sample is:

```
x̂0 = ( xt - √(1-ᾱt) · εθ(xt, t) ) / √ᾱt
```

Setting `σt = 0` for all `t` yields a fully deterministic sampling trajectory, which allows skipping steps (sampling on a subsequence of the diffusion timesteps) and dramatically reduces the number of function evaluations needed at inference time, without retraining the model.

---

### 3. Classifier-Free Guidance (CFG)

CFG enables conditional generation by jointly training a single network on both conditional and unconditional objectives — randomly dropping the conditioning signal `c` (e.g. replacing it with a null token ∅) with some probability during training:

```
εθ(xt, t, c)      — conditional prediction
εθ(xt, t, ∅)      — unconditional prediction
```

At sampling time, the two predictions are linearly combined to steer generation more strongly toward the condition, controlled by guidance scale `w`:

```
ε̂θ(xt, t, c) = εθ(xt, t, ∅) + w · ( εθ(xt, t, c) - εθ(xt, t, ∅) )
```

`w = 0` recovers unconditional generation, `w = 1` recovers standard conditional generation, and `w > 1` amplifies adherence to the condition at some cost to sample diversity — with no separate classifier network required.

---

### 4. Latent Diffusion Model (LDM)

LDM moves the diffusion process from pixel space into a lower-dimensional latent space, greatly reducing computational cost while preserving perceptual quality.

**Autoencoder.** An encoder `E` compresses an image `x` into a latent `z = E(x)`, and a decoder `D` reconstructs it, `x̂ = D(z)`, trained with a perceptual + adversarial objective so the latent space stays perceptually meaningful.

**Latent-space diffusion.** The forward/reverse diffusion process (as in DDPM/DDIM above) is applied entirely to `z` instead of `x`:

```
zt = √ᾱt · z0 + √(1-ᾱt) · ε
L_LDM = E[t, z0, ε] [ || ε - εθ(zt, t) ||² ]
```

**Conditioning.** Cross-attention layers inject conditioning information (text, class labels, etc.) into the denoising network at each resolution level, via a domain-specific encoder `τθ(c)`:

```
Attention(Q, K, V) = softmax( QKᵀ / √d ) · V
Q = WQ · φ(zt),   K = WK · τθ(c),   V = WV · τθ(c)
```

**Decoding.** The final denoised latent `z0` is mapped back to pixel space via the decoder: `x0 = D(z0)`.

---



## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. arXiv:2010.02502.
- Ho, J., & Salimans, T. (2021). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.

