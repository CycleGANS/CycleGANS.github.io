---
layout:     notebook
title:      Discriminator Networks of CycleGANs
author:     Ziyu Zhou
tags:       GANS CycleGANs PatchGANs Discriminator DeepNets Vision
subtitle:   Basic Idea and Implementation of A Simplified Discriminator of CycleGANs
category:  project1
---

# The Discriminator Networks

## Basic Idea

The CycleGAN paper uses the architecture of $70 \times 70$ PatchGANs introduced in paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) for its discriminator networks. The experimental results show that PatchGANs can produce high quality results even with a relatively small patch size


## PatchGANs

The idea of PatchGANs is to split the raw input image into some local small patches, run a general discriminator convolutionally on every patch, and average all the responses to obtain the final output indicating whether the input image is fake or not. 

The main difference between a PatchGAN and a regular GAN discriminator is that the latter maps an input image to a single scalar output in the range of $[0, 1]$, indicating the probability of the image being real or fake, while PatchGAN provides an array as the output with each entry signifying whether its corresponding patch is real or fake.

According to paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf), using a PatchGAN is sufficient because the problem of bulrry images caused by failures at high frequencies like edges and details can be alleviatd by restricting the GAN discriminator to only model high frequencies, and PatchGAN is designed to stress this.

The reason why the CycleGAN paper implements PatchGAN as its discriminator is that it has fewer parameters than a full-image discriminator, and thus runs very fast, being able to work on arbitrarily large images.


## Our Implementation

At this point, we implemented a simplified CycleGAN discriminator, which is a network of 5 convolution layers ([_Figure 1_](#discrim)), including:

* 4 layers to extract features from the image, and
* 1 layer to produce the output (whether the image is fake or not).

We haven't included the structure of PatchGAN at this point. We plan to do it after testing the performance of this simplified version. To further understand how the PatchGAN works, we may use our current implementation as a baseline and test them on more datasets if time allowed.

<center>
![img](img/discriminator.svg)

_Figure 1: Simplified Discriminator Architecture_<a id="discrim"></a>
</center>



## Hyperparameters

The main hyperparameters for the discriminator are, namely, number of output filters, kernel size and stride. A trivial configuration is shown in [Table 1](#table_1). Futher tuning is needed when training the model.

<center>

_Table 1: Hyperparameters_<a id=""table_1></a>

| Layer | Number of output filters | Kernel size | Stride |
|:-----:|:------------------------:|:-----------:|:------:|
|   1   |            64            |     4*4     |    2   |
|   2   |         64*2=128         |     4*4     |    2   |
|   3   |         64*4=256         |     4*4     |    2   |
|   4   |         64*8=512         |     4*4     |    1   |
|   5   |             1            |     4*4     |    1   |

</center>

We also use padding to maintain the informaiton of pixels on the boundary of the image.

## Code Snippet

A code snippet for our simplified discriminator is shown below.

> **Note:** Codes are inspired by the cool works of [leehomyc](https://github.com/leehomyc/cyclegan-1) and [hardikbansal](https://github.com/hardikbansal/CycleGAN).

```python
# General convolution layer.
def conv2d_layer(inputconv, num_filter=64, filter_h=7, filter_w=7, stride_h=1, stride_w=1, stddev=0.02, 
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        
        conv = tf.contrib.layers.conv2d(inputconv, num_filter, filter_h, stride_h, padding, activation_fn=None, 
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        
        if do_norm:
            conv = instance_norm(conv)
            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

# Build model: simplified discriminator
def build_gen_discriminator(input_src, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        layer1 = conv2d_layer(input_src, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        layer2 = conv2d_layer(layer1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        layer3 = conv2d_layer(layer2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        layer4 = conv2d_layer(layer3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        layer5 = conv2d_layer(layer4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return layer5
```

## Sources

1.  [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
2.  [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
3.  [Image-to-Image Translation with Conditional Adversarial Nets (UPC Reading Group)](https://www.slideshare.net/xavigiro/imagetoimage-translation-with-conditional-adversarial-nets-upc-reading-group)
3.  [Notes on the Pix2Pix (pixel-level image-to-image translation) Arxiv paper](https://gist.github.com/brannondorsey/fb075aac4d5423a75f57fbf7ccc12124)
4.  [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)