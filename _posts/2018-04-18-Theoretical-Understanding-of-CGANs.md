---
layout:     notebook
title:      Theoretical Understanding of CGANs 
author:     Harshad Rai
tags:       CycleGANS GettingStarted ComputerVision UnpairedImage2Image ImagetoImage Image2Image Theory
subtitle:   Understanding Cyclce GANs
category:  project1
---


## Introduction to Cycle GANs
Now that we have an idea of Generative Adversarial Networks, we can dive into the heart of this project, i.e. <b>Cycle GANs</b>.  

> **NOTE**: As always, we will be building up the concept of cycle GAN on the previous blogs. Please visit them in order to understand the underlying principles and additional concepts needed to understand this blog. 



Cycle GANs was introduced by Jun-Yan Zhu et. al. in their 2017 paper "[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)" 
They also have an amazing [website](https://junyanz.github.io/CycleGAN/) that provides examples of their outputs, news articles and links to the implementation of their algorithm in different programming languages.



<img src= "{{ "/img/Harshad/UnderstandingCGANs/fullnetwork.png" | prepend: site.baseurl }}">

<center><em>Figure 1: Full Network Flow Diagram</em></center>

The architecture introduced in this paper learns a mapping function $G: X→Y$ using an adversarial loss such that $G(X)$ cannot be distinguished from $Y$, where $X$ and $Y$ are the input and output images respectively.   The algorithm also learns an inverse mapping function $F: Y→X$ using a cycle consistency loss such that $F(G(X))$ is indistinguishable from X. Thus the architecture contains `two Generators` and `two Discriminators`. However, the major aspect in which this implementation truly shines is that it does not require the $X$ and $Y$ pairs to exist, i.e. image pairs are not needed to train this model.  This is highly beneficial as such pairs are not necessarily always available or tend to be expensive monetarily.  
An application of this could be used in movies, where, if a movie crew was unable to shoot a scene at a particular location during the summer season and it is now winter, the movie crew can now shoot the scene and use this algorithm to generate scenes which look like they were shot during the summer. Other areas in which this algorithm can be applied include image enhancement, image generation from sketches or paintings, object transfiguration, etc. The algorithm proves to be superior to several prior methods.



## Detailed overview

Lets dive right into the theory of this paper to understand what exactly happens under the hood.

>  **Goal**: Learn mapping functions between 2 domains $X$ and $Y$.  

To summarize the above TL;DR paragraph, we have :

<b>Training Examples</b>: 
<ul>
    <li> <b>$\{x_i\}_{i=1}^N$</b> where <b>$x_i \in X$</b> </li>
    <li> <b>$\{y_i\}_{i=1}^N$</b> where <b>$y_j \in Y$</b> </li>
    </ul>
<b>Data Distribution</b>: 
<ul>
    <li> <b>$x \sim p_{data}(x)$</b> </li>
    <li> <b>$y \sim p_{data}(y)$</b> </li>
    </ul>
<b>Mappings</b>:
<ul>
    <li> <b>$G: X→Y$</b> </li>
    <li> <b>$F: Y→X$</b> </li>
    </ul>
<b>Discriminators</b>:
<ul>
    <li><b>$D_X→$</b> aims to distinguish between images <b>$\{x\}$</b> and translated images <b>$F(y)$</b> </li>
    <li><b>$D_Y→$</b> aims to distinguish between images <b>$\{y\}$</b> and translated images <b>$G(x)$</b> </li>
    </ul>



### Objective

The Objective of this architecture contains two kinds of losses: 

One for matching the distribution of generated images to the data distribution in the target domain, called as *Adversarial Loss*. The other is to prevent the learned mappings $G$ and $F$ from contradicting each other, called as *Cycle Consistency Loss*.



#### Adversarial Losses:

Adversarial losses need to be applied to both mapping functions. Where generator $G$ tries to generate images that look similar to images from domain $Y$ and discriminator $D_Y$ aims to distinguish between translated samples $G(x)$ and real samples $y$. Hence, $D_Y$ tries to maximize and $G$ tries to minimize the below loss. 

$L_{GAN}(G,D_Y,X,Y)= E_{y \sim p_{data}(y)}[\log(D_Y(y))] + E_{x \sim p_{data}(x)}[\log(1-D_Y( G( x ) ) )]$ 



Similarly, the second adversarial loss for mapping function $F: Y \mapsto X$ and it's discriminator $D_X$ the generator $F$ tries to minimize and discriminator $D_X$ tries to maximize the below loss.

$L_{GAN}(F,D_X,X,Y)= E_{x \sim p_{data}(x)}[\log(D_X(x))] + E_{y \sim p_{data}(y)}[\log(1-D_X( F( y ) ) )]$



#### Cycle Consistency Loss:

In theory, adversarial training learns stochastic mapping functions $G$ and $F$ that produce outputs that are identically distributed as their target domains $Y$ and $X$.
However, with large enough capacity, these functions can map the same set of images to any random permutation of images in the target domain which have the same distribution as the target distribution. This basically means that using only the adversarial losses trains the generators to generate images that look like images from the target set but may not have the same structure as the input images.  

Thus, forward cycle consistency and backward cycle consistency are needed as given below respectively,

$$ x \mapsto G(x) \mapsto F(G(x)) \approx x$$

$$ y \mapsto F(y) \mapsto G(F(y)) \approx y$$



Incentivizing this behavior using *cycle consistency loss*
$$L_{Cyc}(G,F) = E_{x \sim p_{data}(x)}[||F(G(x))-x||_1] + E_{y \sim p_{data}(y)}[||G(F(y))-y||_1]$$



This will result in full objective loss as given below, which we aim to solve by minimizing over each generator i.e. $G$ & $F$ and maximizing over each discriminator i.e. $D_X$ & $D_Y$.

$$L(G,F,D_X,D_Y) = L_{GAN}(G,D_Y,X,Y) + L_{GAN}(F,D_X,Y,X) + \lambda L_{Cyc}(G,F)$$

where $\lambda$ controls relative importance of the two objectives.



### Final thoughts

We have discussed the overall flow of the network with some theoretical concept behind the model formulation. The punch line is that there are two generative adversarial networks binded with a cyclic loss function. The over-all objective function of the network contain both cyclic loss as well as adversarial losses from each GAN. The idea would become more apparent when we start implementing our own Cycle GAN network for image-to-image translation. 



### Sources

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Cycle GAN for image2image translation repository](https://junyanz.github.io/CycleGAN/)