---
marp: true
title: Exploring Autoencoders
author: George Brova
date: 2024-01-19
# theme: uncover
class: invert
---

# Exploring Autoencoders

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![width:400px center autoencoder example](https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png)

George Brova

---

## Basic example

```python
encoder = tf.keras.Sequential([
    Input(shape=(inferred_size, inferred_size, 3)),
    Flatten(),
    Dense(192, activation='relu')  # 192 = 8*8*3
    
])

decoder = tf.keras.Sequential([
    Dense(32*32*3, activation='sigmoid'),
    Reshape((32, 32, 3)),
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(x_train, x_train, ...)

```

---

## What this does

Original 32x32x3 images (total 3072 numbers):
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-originals.png?raw=true)

Reconstruction with a 192-d bottleneck:

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-deep-reconstructions.png?raw=true)

---

## Is that any good?

- Baseline: no fancy ML, just downsample the image!
- 192-d bottleneck is equivalent to 8x8x3, so downsample 32x32 -> 8x8.

Baseline: Downsample-and-reupsample (MSE=0.022):
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-downsample-baseline.png?raw=true)

Our AE, again (MSE=0.014):
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-deep-reconstructions.png?raw=true)

---

## Convolutions - motivation

- That simple dense NN has ~1.2M parameters!
- This won't scale for deeper NNs (more layers)
- This won't scale for larger images (N x N images -> O(N^2) params)

---

## Convolutions - code

```python3
encoder = tf.keras.Sequential([
    Input(shape=(inferred_size, inferred_size, 3)),

    Conv2D(32,kernel_size=3,activation='relu',padding='same',strides=1),
    MaxPooling2D((2, 2), padding='same'),
    BatchNormalization(),
    Conv2D(64,kernel_size=3,activation='relu',padding='same',strides=1),
    MaxPooling2D((2, 2), padding='same'),
    BatchNormalization(),
    Conv2D(12,kernel_size=3,activation='relu',padding='same',strides=1), 
    MaxPooling2D((2, 2), padding='same'),
])

decoder = tf.keras.Sequential([
    UpSampling2D((2, 2)),
    Conv2D(64,kernel_size=3,strides=1,activation='relu',padding='same'),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(32,kernel_size=3,strides=1,activation='relu',padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(3,kernel_size=(3,3),activation='sigmoid',padding='same')
])

autoencoder = tf.keras.Sequential([encoder, decoder])
```

---

## Convolutions - results

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-reconstructions.png?raw=true)

- ~ same reconstruction error.
- But only 50k parameters this time!

---

## Which reconstruction is better?

1:
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-reconstructions.png?raw=true)

2:
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-downsample-baseline.png?raw=true)

---

## Which reconstruction is better? (2)

1: (MSE=0.013)
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-reconstructions.png?raw=true)

2: (MSE=0.022)
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-downsample-baseline.png?raw=true)

---

## Loss functions

### What's going on?

- MSE loss compares each pixel brightness independently and aggregates the differences.
- But human vision focuses on shapes and contours.

---

## Loss functions - SSIM

Same architecture, change loss function:

```python3
...

autoencoder.compile(optimizer='adam', loss=SSIMLoss)
```

Result:

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-ssim-reconstructions.png?raw=true)

---

## Loss functions - side-by-side comparison

SSIM loss:
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-ssim-reconstructions.png?raw=true)

MSE loss:
![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/32x32-emoji-conv-reconstructions.png?raw=true)

---

## Denoising

Generate synthetic noise (like film grain):

```python3
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
```

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/128x128-emoji-noisy.png?raw=true)

---

## Denoising - recovery (naive)

```python3
autoencoder.fit(x_train, x_train, ...)
```

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/128x128-emoji-noisy-recovered.png?raw=true)

---

## Denoising - recovery (retrained, explicit)

```python3
autoencoder.fit(x_train_noisy, x_train, ...)
```

![](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/128x128-emoji-noisy-recovered-v2.png?raw=true)

---

## Anomaly detection

Intuition:

1. Train an autoencoder on some regular (not-anomalous) data.
1. To decide if some new data is anomalous, calculate its reconstruction loss.
1. If the reconstruction loss is higher than for the original dataset, this new data is anomalous.

---

## Anomaly detection - results

- Train AE on emoji
- Anomalies are MNIST digits
- Can clearly separate the reconstruction losses:

![h:400 center ](https://github.com/gbrova/fun-with-autoencoders/blob/feature/intro-blogpost/screenshots/anomaly-recovery-distribution-emoji-mnist.png?raw=true)

---

## My takeaways so far

- It's satisfying to see why something doesn't work, before figuring out what does.  Many online resources lack this :-(
- Trite but true: it's all in the optimization (MSE vs SSIM; `x_train` vs `_lossy`)

---

## Next steps?

- Similar exploration, for variational autoencoders
  - What happens if I just sample the embedding space of a regular AE?
  - Build the variational tricks, one by one.
- AEs as a way to teach computer vision?
  - Reconstructions can show what structure the model is capable of learning.
- Eyes open for new debugging tricks.

## Pair with me

- On one of these things, or something totally different :-)
