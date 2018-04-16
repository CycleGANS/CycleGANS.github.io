---
layout:     notebook
title:      Theoritical Insights of GANs
author:     Harshad Rai
tags:       GANS GettingStarted ComputerVision Vision Theory
subtitle:   Understanding Generative Adversarial Networks
category:  project1
---

In order to understand CycleGANs, it is important that we understand GANs (Generative Adversarial Networks) first. GANs were introduced by Ian Goodfellow et. al. in their 2014 paper title "Generative Adversarial Nets". The paper can be found [here](https://arxiv.org/pdf/1406.2661.pdf).

## Generative Adversarial Networks
Generative Adversarial Networks aka GANs are feedforward neural networks that create images from random noise that approximate real images.
This is done using two neural networks:
* A Generator
* A Discriminator

Both, the generator and the discriminator, are multilayer perceptrons. An architecture that uses both of them is referred to as <b>Adversarial Nets</b>.
Both the models are trained using backpropagation and dropout algorithms and samples obtained from the generator only using forward propagation.
Adversarial nets is most straightforward to apply when both models are multilayer perceptrons.

To understand this a little more in depth, let us look at a few notations and understand what each model is attempting to do in order to acheive the results obtained from GANs
<b>$p_g$</b> = Generator's Distribution over data <b>$x$</b>
<b>$p_z(z)$</b> = Prior on input noise variables
<b>$G(z;\theta_g)$</b> = Mapping to data space where <b>$G$</b> is a differentiable function represented by a multilayer perceptron with parameters <b>$\theta_g$</b>
<b>$D(x,\theta_d$</b> = Second multilayer perceptron (Discriminator) that outputs a single scalar
<b>$D(x)$</b> = Probability that <b>$x$</b> came from data rather than <b>$p_g$</b>

The task for the Generator network is to approximate a function <b>$G(z;\theta_g)$</b> that maps random noise to a range whose probability distribution <b>$p_g$</b> is the same as the probability distribution of the real data <b>$x$</b>.
While the discriminator is tasked with differentiating images coming out of the Generator and real data.

#### Training
* <b>$D$</b> is trained to maximize the probability of assigning the correct label to both: traning examples and samples from <b>$G$</b>
* Simultaneously, <b>$G$</b> is trained to <b>$Minimize\log(1-D( G( z ) ) )$</b>
##### Understanding this:
<b>$G(z)$</b> is a fake image.
<b>$D(G(z))$</b> is the probability of discriminator classifying this fake image as true data.
<b>$G$</b> would like to maximize this.
i.e., if <b>$D(G(z)) = 1$</b>, then  <b>$\log(1-D( G( z ) ) )=\log(1-1)=-\infty$</b>
but if <b>$D(G(z)) = 0$</b>, then <b>$\log(1-D( G( z ) ) )=\log(1-0)=0$</b>

In other words, <b>$D$</b> and <b>$G$</b> play the following two player minimax game with value function <b>$V(D,G)$</b>:
$$Min_G Max_DV(D,G)=E_{x \sim p_{data}(x)}[\log(D(x))]+E_{z \sim p_z(z)}[\log(1-D( G( z ) ) )]$$

In practise, the implementation is carried out in an iterative manner to avoid overfitting and computational prohibition of optimizing <b>$D$</b> to completion in the inner loop of training. Instead, <b>$D$</b> and <b>$G$</b> are optimized alternately with k optimization steps of <b>$D$</b> followed by one optimization step of <b>$G$</b>. This allows <b>$D$</b> to be maintained near its optimal solution as long as <b>$G$</b> chnages slowly.

The algorithm provided in the GANs paper is as follows:

![image.png](https://github.com/CycleGANS/CycleGANS.github.io/blob/master/img/Harshad/image.png?raw=true)

The theory of this paper states that as long as <b>$D$</b> and <b>$G$</b> have enough capacity, <b>$p_g$</b> converges to <b>$p_{data}$</b>
