---
title: "IRMAE"
date: "2026-02-15"
layout: "post"
tags:
    - "machine-learning"
    - "neural-networks"
    - "irmae"
    - "autoencoders"
---


# Implicit Rank-Minimizing Autoencoder (IRMAE)

We know that almost everything neural networks-related operates using tensors and matrices, and this extends to Autoencoders. Clearly, from the name, you can tell an IRMAE minimizes the rank of your Autoencoder, implicitly. But what exactly does that mean, what does that do, why do you want to do it, and how is that implemented, are all questions you'll hopefully not be having by the end of this blog. 

## What an IRMAE does differently from an AE

![alt text](/assets/images/AE.png)

In your typical Autoencoder (AE), you have 3 main parts. You have an input layer, which is the data you want to encode, as you can see from the image, you decrease the dimensions of the data as you go through the layers. This essentially compresses the input data into a lower-dimensional representation. You then have your Latent Space, which is the bottleneck layer that contains the compressed representation of the input data. Finally, you have your output layer, which is where the model tries to reconstruct the original input data from the compressed representation in the latent space.

A simpler diagram would be:

![alt text](/assets/images/AE-sketch.png)

Now, your IRMAE is almost the same thing, but you add extra layers between the encoder and the latent space. 

![alt text](/assets/images/IRMAE-sketch.png)


### What these extra layers do

It reduces the rank (NOT to be confused with size/dimensions reduction)

In your normal AE, the rank of the latent space is equal to the number of neurons, and the size reduces as your latent code is a bottleneck. So if your latent code has, for example, 16 dimensions, then the rank of the latent space is 16, and all 16 dimensions are used to encode the data.

In an IRMAE, the extra layers thin out the information. The rank of the latent space is reduced by adding extra layers between the encoder and the latent space. This means that even if your latent code has 16 dimensions, the extra layers will force the model to use fewer dimensions, for example, 8 dimensions.

Now you might think- Why don't i just use a smaller latent code of like 8 dimensions? Its because you don't know the true dimensionality of the data. If you use a smaller latent code, you might be throwing away important information. If you use a larger latent code, you'd be overfitting. With IRMAE though, you can use a larger latent code and let the model figure out how many dimensions to use by reducing the rank. 


### What happens to the layers

In practice, they are absorbed into the last layer of the encoder during inference. These matrices- (W1, W2, ....) are equivalent to a single linear layer at inference time, so they do not change the capacity of the autoencoder. 

![alt text](/assets/images/IRMAE-layers.png)

#### How are the layers 'absorbed'?

Typically, in a normal AE, your L2 reconstruction loss is 
![alt text](/assets/images/L2-loss.png)
where y is your input, ε() is your encoder, and D() is your decoder.

In an IRMAE, your L2 reconstruction loss is
![alt text](/assets/images/IRMAE-loss.png)
As you can tell, the extra layers are just matrices that get “absorbed” into the encoder when all the linear matrices
collapse, as linear matrix multiplication is associative.

## Why do you want IRMAE?

In normal AEs, you have a problem. If you don’t constrain them properly, they can just learn to copy the input instead of learning anything meaningful. Your latent space gets messy, high-dimensional, and can have holes, which makes it bad for things like generation and interpolation.


In IRMAEs, you implicitly push the latent space towards a lower rank, so you get a more compact latent space that still works well for reconstruction, generation. 

You COULD try fixing this by adding explicit constraints, eg.  reducing the latent size, adding noise (VAE), or enforcing sparsity. But these have their own problems, eg. you have to guess the right latent size, it can be comparatively noisier, etc.

## Implementations

To show how effective IRMAEs are, we check whether the model can automatically discover the true number of underlying factors in the data by analyzing how many latent dimensions it actually uses. We compare the performance of a normal AE, an AE with L1 and L2 regularization, and an IRMAE.

Here, we use a synthetic dataset, in which each image is a 32x32 RGB image with a random color, size, shape, and position. From this, we can tell that the intrinsic dimensionality is 7 (3 for color, 2 for position, and 1 for size and one for shape). We want our model to use just 7 dimensions to encode the data, even though the latent code has 32 dimensions.
![alt text](/assets/images/Exp4-1_graph.png)

From the result, you can see that the IRMAE is actually minimizing the rank, only the first few singular values are non-zero, which means that the model is using only a few dimensions in the latent space to encode the data. The L1 and L2 regularized AEs DO regularize, but singular values reduce slowly, and there are many dimensions still active.
