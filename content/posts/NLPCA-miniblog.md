---
title: "Understanding Sequential NLPCA"
date: "2026-02-15"
layout: "post"
tags:
    - "machine-learning"
    - "neural-networks"
    - "nlpca"
    - "pca"
---

Maybe it's just me, but a lot of the times when reading a research paper, everything written just seems to fly our the top of my head, despite multiple reads. Is it the concept itself being hard to grasp and me being intimidated by the formatting? Or is it just me being dense? (most likely)
Whatever it is, I decided to make a mini-blog that's much simpler to understand, and a little easier on the eyes.

## What problem are we solving using NLPCA

In a normal PCA (Principal Component Analysis), we compress data by finding linear combinations of variables that capture the most variation. It finds LINEAR combinations, so it can only handle linear relationships, but what if the data lies on a curve? Instead of projecting data onto straight lines like PCA, NLPCA uses a neural network to learn a nonlinear compression.

## The Basic idea of NLPCA

- Build a neural network that tries to reproduce its input.
- Make it pass through a bottleneck i.e. small hidden layer.
- bottleneck has fewer neurons and dimensions than the input, you essentially force the network to compress the data.

This looks like:

![NLPCA layer architecture](/assets/images/NLPCA-layer.png)

So you want the output to be the same as input, and when this happens, the bottleneck layer basically contains a compressed representation of the data, which is the nonlinear principal components.


## How Multiple Nonlinear Components are extracted

In the paper, there's two different strategies. 
- Simultaneous NLPCA: Use multiple bottleneck neurons at once and train everything together.
- Sequential NLPCA: Extract one nonlinear component at a time, just like how PCA can be done sequentially.

I found the sequential approach much easier to reason about, so here we are :D 

## Sequential NLPCA: How It Works

Instead of training one big network with multiple bottleneck neurons, you extract the First Nonlinear Component, build a network with only one bottleneck neuron, and train it to reproduce the input. That bottleneck neuron now represents the first nonlinear principal component.

Ofcourse, since its a bottleneck, there will be some data that the first nonlinear component (i.e. the bottleneck) failed to capture. Lets call this residual. You feed this residual into a new network (which also has a single bottleneck neuron). You train it to reconstruct the residual. This gives you the second nonlinear component. You repeat this as many times as you need to.

I like to relate it to the Matryoshka dolls (the stacking dolls, where you find a smaller doll inside the bigger one). Each network removes one layer of nonlinear structure. What’s left becomes simpler and simpler i.e. smaller and smaller.

## Similarity with PCA

In classical PCA, the first principal component explains the most variance, the second is the first principal component of the residual, and so on...

Sequential NLPCA mimics this exact idea, but instead of linear projections, we’re learning nonlinear mappings with neural networks.

## Sequential NLPCA vs Simultaneous NLPCA

Here's some instances when one of them can be better than the other

- When Sequential>> Simultaneous

    - When we train multiple bottleneck neurons at once, they can accidentally learn the same thing, this usually happens with dominant features. In sequential on the other hand, each network only learns one component, the next network's input is what the first network didn't grasp, so this duplication problem doesn’t happen.

    - Better Training Stability: After removing the first major nonlinear structure, the remaining residual is smaller. You can even rescale it before training the next network, this makes training easier and more stable.

- When Sequential<< Simultaneous
    - When Nonlinear Components Are Strongly Interdependent. When its sequential, the thought process is kinda like "After removing component 1, the rest of the structure is still meaningful on its own." This is, a lot of the times, not the case.
    - When your data isn't noisy and your optimization method is strong, Simultaneous training usually performs better.