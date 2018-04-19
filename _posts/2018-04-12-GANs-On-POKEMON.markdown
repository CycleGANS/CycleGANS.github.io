---
layout:     notebook
title:      GANs On Pokemon
author:     Naman Shukla
tags:       GANS GettingStarted DeepNets Vision Pokemon
subtitle:   Generating New Pokemons with GANs
category:  project1
---

# Training Pokemon with GANs

> **Note:** Special thanks to [Zhenye Na](https://github.com/Zhenye-Na/) from helping us on this part of the project. We would like to thank [Siraj Raval](https://www.linkedin.com/in/sirajraval/) for the [video](https://youtu.be/yz6dNf7X7SA) and repository contribution. 



Generating Pokemon from `GANs` seems really interesting! So decided to implement this for fun. The neural network architecture that we have used for training Pokemon is [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434) (aka `DCGAN`) 



### About Discriminator

In `DCGAN` architecture, the discriminator `D` is Convolutional Neural Networks (`CNN`) that applies a lot of filters to extract various features from an image. The discriminator network will be trained to discriminate between the original and generated image. The process of convolution is shown in the illustration below :



![img](https://camo.githubusercontent.com/87c865a3c5894d14b98b36b647da3f67e1bd166c/687474703a2f2f646565706c6561726e696e672e6e65742f736f6674776172652f746865616e6f5f76657273696f6e732f6465762f5f696d616765732f73616d655f70616464696e675f6e6f5f737472696465735f7472616e73706f7365642e676966)

### Overview of the network architecture for Discriminator

The high level pipeline for this implementation is given as follows: 

| Layer       | Shape                   | Activation |
| ----------- | ----------------------- | ---------- |
| input       | batch size, 3, 64, 64   |            |
| convolution | batch size, 64, 32, 32  | LRelu      |
| convolution | batch size, 128, 16, 16 | LRelu      |
| convolution | batch size, 256, 8, 8   | LRelu      |
| convolution | batch size, 512, 4, 4   | LRelu      |
| dense       | batch size, 64, 32, 32  | Sigmoid    |

#### Code

```python
def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        # Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(
            input, c2, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv1')
        bn1 = tf.contrib.layers.batch_norm(
            conv1,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn1')
        act1 = lrelu(conv1, n='act1')
        # Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(
            act1, c4, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(
            conv2,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn2')
        act2 = lrelu(bn2, n='act2')
        # Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(
            act2, c8, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(
            conv3,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn3')
        act3 = lrelu(bn3, n='act3')
        # Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(
            act3, c16, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(
            conv4,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2',
                             shape=[fc1.shape[-1],
                                    1],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
    return logits  # , acted_out
```



### About Generator

The generator G, which is trained to generate image to fool the discriminator, is trained to generate image from a random input. In DCGAN architecture, the generator is represented by convolution networks that upsample the input. The goal is to process the small input and make an output that is bigger than the input. It works by expanding the input to have zero in-between and then do the convolution process over this expanded area. The convolution over this area will result in larger input for the next layer. The process of upsampling is shown below:

![img](https://camo.githubusercontent.com/7a8ee405b6f08ac12d0754511dd79bee457dc438/687474703a2f2f646565706c6561726e696e672e6e65742f736f6674776172652f746865616e6f5f76657273696f6e732f6465762f5f696d616765732f70616464696e675f737472696465735f7472616e73706f7365642e676966)

Depending on sources, you can find various annotations for the upsample process. Sometimes they are referred as full convnets, in-network upsampling, fractionally-strided convolution, deconvolution and it goes on and on. 



### Overview of the network architecture for Generator

The high level pipeline for this implementation is given as follows: 

| Layer         | Shape                                             | Activation |
| ------------- | ------------------------------------------------- | ---------- |
| input         | batch size, 100 (Noise from uniform distribution) |            |
| reshape layer | batch size, 100, 1, 1                             | Relu       |
| deconvolution | batch size, 512, 4, 4                             | Relu       |
| deconvolution | batch size, 256, 8, 8                             | Relu       |
| deconvolution | batch size, 128, 16, 16                           | Relu       |
| deconvolution | batch size, 64, 32, 32                            | Relu       |
| deconvolution | batch size, 3, 64, 64                             | Tanh       |

#### Code

```python
def generator(input, random_dim, is_train, CHANNEL, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32  # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable(
            'w1',
            shape=[
                random_dim,
                s4 * s4 * c4],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(
            conv1,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        # Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(
            act1, c8, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(
            conv2,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(
            act2, c16, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(
            conv3,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(
            act3, c32, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(
            conv4,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(
            act4, c64, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv5')
        bn5 = tf.contrib.layers.batch_norm(
            conv5,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        # 128*128*3
        conv6 = tf.layers.conv2d_transpose(
            act5, output_dim, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
    return act6
```



### Hyperparameter of DCGAN

One thing that everyone notices is that the GANs are highly computationally expensive. The thing that people overlook generally is how fragile GANs are with respect to hyperparameters. GANs work exceptionally well with certain parameters but not with others. Currently tunning these knobs are part of the art in designing network architecture. The hyperparameteres that we have decided to go with are as follows:

| Hyperparameter                                               |
| ------------------------------------------------------------ |
| Mini-batch size of 64                                        |
| Weight initialize from normal distribution with std = 0.02   |
| LRelu slope = 0.2                                            |
| Adam Optimizer with learning rate = 0.0002 and momentum = 0.5 |

### Pokemon Dataset

We have used a compiled data set of 871 different Pokemon (different generation of the pokemons) available at [moxiegushi](https://github.com/moxiegushi/pokeGAN). This data set is compiled from [Kaggle competition data set](https://www.kaggle.com/dollarakshay/pokemon-images/discussion) and [PokeDex dataset](https://veekun.com/dex/downloads). 

All images will be reshaped to 64x64 pixels with white background. If an image is in png format and has a transparent background (i.e. RGBA), it will be converted to jpg format with RGB channel.



### Implementation

The entire for training the pokemons are linked in our repository. This repository is inspired from  [Newmu](https://github.com/Newmu/dcgan_code), [kvpratama](https://github.com/kvpratama/gan/tree/master/pokemon), [moxiegushi](https://github.com/moxiegushi/pokeGAN) along with our own implementation.

#### Dependencies 

> **Note** : The following packages must be installed in your machine if you want to run pokemon-gan : 
>
> 1. scikit-image
> 2. tensorflow
> 3. scipy
> 4. numpy
> 5. Pillow

The dependencies can be installed by following the commands :

```
git clone https://github.com/Zhenye-Na/pokemon-gan.git
cd pokemon-gan  
pip install -r requirements.txt
```



#### Running the code

> **Note** : Please note that running GANs is computationally expensive process. Hence, we recommend using GPU or cloud servers and backing up data for running on CPU.

You can run our code by following commands:

```
git clone https://github.com/Zhenye-Na/pokemon-gan.git
cd pokemon-gan
python3 main.py
```



### Results

> **Note** : Currently we are running our code on CPU. So we don't have full results (only 250 epochs out of 1000 epochs). The code is still running while you are reading this post. We will update the results soon. Stay tuned !

<img src= "{{ "/img/Naman/pokemon/750EP.gif" | prepend: site.baseurl }}">

Output after 800 Epochs:

<img src= "{{ "/img/Naman/pokemon/epoch800.jpg" | prepend: site.baseurl }}">



### Sources

1.  [Generative Adversarial Networks for beginners - Oreilly](https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners)

2. [Introductory guide to Generative Adversarial Networks (GANs) and their promise!](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)

3. [A (Very) Gentle Introduction to Generative Adversarial Networks (a.k.a GANs)](https://www.slideshare.net/ThomasDaSilvaPaula/a-very-gentle-introduction-to-generative-adversarial-networks-aka-gans-71614428)

4. [Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)

5. [An introduction to Generative Adversarial Networks (with code in TensorFlow)](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

6. [Generative Adversarial Networks Explained with a Classic Spongebob Squarepants Episode](https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39)

7. [UCLA ACM AI Generative-Adversarial-Network-Tutorial](https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial)

   ​