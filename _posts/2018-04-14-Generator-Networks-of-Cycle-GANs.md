---
layout:     notebook
title:      Generator Networks of Cycle GANs
author:     Harshad Rai
tags:       GANS CycleGANs ComputerVision Vision Theory Image2Image NeuralNetwork DeepLearning
subtitle:   Understanding and Building the Generator Network of CycleGANs
category:  project1
---


## Generator Network

Now that we have understood the theory behind Cycle GANs, it is time for us to look at the implementation. In this article, we will have a look at the Generator architecture used in the CycleGANs [paper](https://arxiv.org/pdf/1703.10593.pdf).

In the implementation section of this paper, the authors state that the Generator architecture used, was obtained from the paper titled [Perceptual Losses for Real Time Style Transfer](https://arxiv.org/abs/1603.08155) by Justin Johnson et. al.

Justin Johnson et. al. utilize perceptual losses to perform style transforms between two set of images. They also perform single image super resolution. They claim that their implementation provides similar results when compared to optimization models while being three orders in magnitude faster. Now that's an amazing feat!  
They state that their architecture does not rely only on pixel losses but also on perceptual losses which is the most imporant as small translations and rotations of the image give large pixel losses which is not gthe case when considering perceptual losses. This results in absolutely stunning outputs as shown in their paper.  
However, we shall not get into the perceptual losses and their benefits in this article as that is not our major goal. Our motive to understand their architecture is limited to their Generator architecture which generates the amazing outputs and also does this at a really fast pace.

The Generator used in this implementation involves three parts to it:
<ul>
    <li> <b>In-network Downsampling</b> </li>
    <li> <b>Several Residual Blocks</b> </li>
    <li> <b>In-network Upsampling</b> </li>
    </ul>
    

<b>In-Network Downsampling</b>  
<ul>
    This part of the Generator consists of two **[Convolution Networks](https://cyclegans.github.io/project1/2018/04/04/Getting-Started-With-CNN/)**, each followed by **[Spatial Batch Normalization](https://www.youtube.com/watch?v=vq2nnJ4g6N0&amp;t=76m43s)** and a ReLu activation function.  
    Each convolution network uses a stride of 2 so that downsampling can occur.  
    The first layer has a kernel size of 9x9 while the second layer has a kernel size of 3x3.
    </ul>
<b>Residual Blocks</b>
<ul>
    The concept of Residual Blocks was introduced by Kaiming He et. al. in their paper titled [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).  
    Each Residual Block consists of two Convolution Layers. The first Convolution Layer is followed by Batch Normalization and ReLu activation. The output is then passed through a second Convolution Layer followed by Batch Normalization. The output obtained from this is then added to the original input.  
    <img src= "{{ "/img/Harshad/ResidualBlock.png " | prepend: site.baseurl }}" >
    To understand this in more depth, you can look at this [blog](http://torch.ch/blog/2016/02/04/resnets.html) where the above image was taken from.  
    Each convolution layer in residual blocks has a 3x3 filter.  
    The number of Residual Blocks depens on the size of the input image. For 128x128 images, 6 residual blocks are used and for 256x256 and higher dimensional images, 9 residual blocks are used.
    </ul>
<b>In-network Upsampling</b>
<ul>
    This part consists of two convolution layers.  
    They are fractionally strided with a stride value of $\frac{1}{2}$
    The first convolution layer has a kernel size of 9x9 while the last layer has a kernel size of 9x9.  
    The first layer is followed by Batch Normalization and ReLU activation while the second convolution layer is followed by a Scaled Tanh function so that the values can fall between [0,255] as this layer is the output layer.
    </ul>
The entire Feedforward Generator Network starts off with Downsampling, followed by Residual Blocks and ends with Upsampling.
<img src= "{{ "/img/Harshad/Generator.png " | prepend: site.baseurl }}" style="width: x%; margin-left: y%; margin-right: z%;">

The benefits of using such a network is that it is computationally less expensive compared to the naive implementation and provides large effective receptive fields that lead to high quality style transfers in the output images.


#### Implementation:
Our implementation of the Generator can be found [here](https://github.com/CycleGANS/CS543CycleGANsProject/blob/master/Generator.py)
