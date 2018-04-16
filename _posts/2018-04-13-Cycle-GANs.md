---
layout:     notebook
title:      Cycle GANs
author:     Harshad Rai
tags:       CycleGANS GettingStarted ComputerVision UnpairedImage2Image ImagetoImage Image2Image
subtitle:   Understanding Cyclce GANs
category:  project1
---


## Cycle GANs
Now that we have an idea of Generative Adversarial Networks, we can dive into the heart of this project, i.e. <b>Cycle GANs</b>.  
Cycle GANs was introduced by Jun-Yan Zhu et. al. in their 2017 paper "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" that can be found [here](https://arxiv.org/abs/1703.10593).  
They also have an amazing [website](https://junyanz.github.io/CycleGAN/) that provides examples of their outputs, news articles and links to the implementation of their algorithm in different programming languages.

The architecture introduced in this paper learns a mapping function <b>$G: X→Y$</b> using an adversarial loss such that <b>$G(X)$</b> cannot be distinguished from <b>$Y$</b>, where <b>$X$</b> and <b>$Y$</b> are the input and output images respectively.   The algorithm also learns an inverse mapping function <b>$F: Y→X$</b> using a cycle consistency loss such that <b>$F(G(X))$</b> is indistinguishable from X. Thus the architecture contains two Generators and two Discriminators. However, the major aspect in which this implementation truly shines is that it does not require the <b>$X$</b> and <b>$Y$</b> pairs to exist, i.e. image pairs are not needed to train this model.  This is highly beneficial as such pairs are not necessarily always available or tend to be expensive monetarily.  
An application of this could be used in movies, where, if a movie crew was unable to shoot a scene at a particular location during the summer season and it is now winter, the movie crew can now shoot the scene and use this algorithm to generate scenes which look like they were shot during the summer. Other areas in which this algorithm can be applied include image enhancement, image generation from sketches or paintings, object transfiguration, etc. The algorithm proves to be superior to several prior methods.

Lets dive right into the theory of this paper to understand what exactly happens under the hood.

<b>Goal</b>: Learn mapping functions between 2 domains <b>$X$</b> and <b>$Y$</b>.  

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

The <b>Objective</b> of this architecture contains two kinds of losses:
<ul>
    <li> <b>Adversarial Losses</b> → for matching the distribution of general images to the data distribution in the target domain.</li>
    <li><b>Cycle Consistency Losses</b> → to prevent the learned mappings <b>$G$</b> and <b>$F$</b> from contradicting each other.
    </ul>
  

##### Adversarial Losses:
<ul>
Adversarial losses need to be applied to both mapping functions.  

For Generator <b>$G: X→Y$</b> and Discriminator <b>$D_Y$</b> 

$$L_{GAN}(G,D_Y,X,Y)=E_{y \sim p_{data}(y)}[\log(D_Y(y))] + E_{x \sim p_{data}(x)}[\log(1-D_Y( G( x ) ) )]$$

<b>$G$</b> tries to generate images that look similar to images from domain <b>$Y$</b>  
<b>$D_Y$</b> aims to distinguish between translated samples <b>$G(x)$</b> and real samples <b>$y$</b>
<ul>
    <li><b>$D_Y ~ maximizes~ L_{GAN}(G, D_Y, X, Y)$</b></li>
    <li><b>$G ~ minimizes ~ L_{GAN}(G,D_Y,X,Y)$</b></li>
    </ul>    
    
Similarly, the second Adversarial Loss for mapping function <b>$F: Y→X$</b> and it's discriminator <b>$D_X$</b>:  
$$L_{GAN}(F,D_X,X,Y)=E_{x \sim p_{data}(x)}[\log(D_X(x))] + E_{y \sim p_{data}(y)}[\log(1-D_X( F( y ) ) )]$$  
and the <b>Objective</b> is:
$$Minimize_FMaximize_{D_X}(F,D_X,Y,X)$$
</ul>


##### Cycle Consistency Loss:
<ul>
<b>Motivation</b>  
In theory, adversarial training learns stochastic mapping functions <b>$G$</b> and <b>$F$</b> that produce outputs that are identically distributed as their target domains <b>$Y$</b> and <b>$X$</b>.
However, with large enough capacity, these functions can map the same set of images to any random permutation of images in the target domain which have the same distribution as the target distribution. This basically means that using only the adversarial losses trains the generators to generate images that look like images from the target set but may not have the same structure as the input images.  

What we need to have is:
$$ x → G(x) → F(G(x)) ≈ x$$ called as <b>Forward Cycle Consistency</b>.

And,
$$ y → F(y) → G(F(y)) ≈ y$$ called as <b>Backward Cycle Consistency</b>.  

Incentivizing this behavior using <b>Cycle Consistency Loss</b>
$$L_{Cyc}(G,F) = E_{x \sim p_{data}(x)}[||F(G(x))-x||_1]+E_{y \sim p_{data}(y)}[||G(F(y))-y||_1]$$

#### Full Objective:
$$L(G,F,D_X,D_Y) = L_{GAN}(G,D_Y,X,Y) + L_{GAN}(F,D_X,Y,X) + \lambda L_{Cyc}(G,F)$$
where $\lambda$ controls relative importance of the two objectives.

<b>Aim to Solve</b>:
$$G^*,F^*=arg~min_{G,F}max_{D_X,D_Y}L(G,F,D_X,D_Y)$$
