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

<img src= "{{ "/img/Harshad/GAN/network.png" | prepend: site.baseurl }}">

<center><em>Figure 1: Network Flow Chart (source: KDnuggets - Al Gharakhanian)</em></center>

To understand this a little more in depth, let us look at a few notations and understand the functions of each component in the entire network. 


> $z$ : latent space
>
> $p_z(z)$ : Prior on input noise variables  
>
> $p_g$ : Generator's Distribution over data $x$  
>
> $G(z;\theta_g)$ : Mapping to data space where $G$ is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$  
>
> $D(x,\theta_d)$ : Second multilayer perceptron (Discriminator) that outputs a single scalar  
>
> $D(x)$ : Probability that $x$ came from data rather than $p_g$

The task for the Generator network is to approximate a function $G(z;\theta_g)$ that maps random noise in latent space to a range whose probability distribution $p_g$ is the same as the probability distribution of the real data $x$.
While the discriminator is tasked with differentiating images coming out of the Generator and real data.

### Training
* $D$ is trained to maximize the probability of assigning the correct label to both: training examples and samples from $G$
* Simultaneously, $G$ is trained to $\min\log(1-D( G( z ) ) )$



#### Whats really happening :

$G(z)$ is a fake image.  
$D(G(z))$ is the probability of discriminator classifying this fake image as true data.  
$G$ would like to maximize this.  

> if $D(G(z)) = 1$, then $\log(1-D( G( z ) ) )=\log(1-1) \rightarrow -\infty$  
>
> if $D(G(z)) = 0$, then $\log(1-D( G( z ) ) )=\log(1-0) \rightarrow 0$



In other words, $D$ and $G$ play the following two player minimax game with value function $V(D,G)$:
$$\underset{G}{\text{min}} \; \underset{D}{\text{max}} \;V(D,G)=E_{x \sim p_{data}(x)}[\log(D(x))]+E_{z \sim p_z(z)}[\log(1-D( G( z ) ) )]$$

In practice, the implementation is carried out in an iterative manner to avoid overfitting and computational prohibition of optimizing $D$ to completion in the inner loop of training. Instead, $D$ and $G$ are optimized alternately with k optimization steps of $D$ followed by one optimization step of $G$. This allows $D$ to be maintained near its optimal solution as long as $G$ changes slowly.

The algorithm provided in the GANs article is as follows:

<img src= "{{ "/img/Harshad/GAN/algo.png" | prepend: site.baseurl }}">

<center><em>Figure 2: GAN Algorithm (source: original article)</em></center>

The theory of this paper states that as long as $D$ and $G$ have enough capacity, $p_g$ converges to $p_{data}$



### Final thoughts

Generative Adversarial Networks are an interesting development, giving us a new way to do unsupervised learning. One big open problem in this area is how best to evaluate these sorts of models. Another thing that we've noticed in our implementation is the [mode collapse](http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/). This will become crystal clear once you read our next blog post where we introduce [minimalistic GANS implementation on MNIST](https://cyclegans.github.io/project1/2018/04/13/Minimal-GAN-Modeling-On-MNIST/)  - with just one hidden layer network. Overall, the generative adversarial network is revolutionary concept which is still evolving in many different domains and we will get to know its strengths and weaknesses as we go deeper (no pun intended). 



### Sources

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434v2)
- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

Feel free to reuse our [GAN code](https://github.com/CycleGANS/V1.0), and of course keep an eye on our [blog](https://cyclegans.github.io). Comments, corrections and feedback are welcome.