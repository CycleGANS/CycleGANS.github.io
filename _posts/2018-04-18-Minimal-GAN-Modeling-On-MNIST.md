---
layout:     notebook
title:      Minimal GAN modeling on MNIST
author:     Naman Shukla
tags:       GANS MNIST DeepNets Vision
subtitle:   A detailed description for building a simple GAN model 
category:  project1
---

## About the MNIST Dataset

The dataset is well known I guess due to great Yann LeCun and all unnecessary information can be found [here](http://yann.lecun.com/exdb/mnist/).  Still if you are wondering about the dataset, here it is :
<img src= "{{ "/img/Naman/GAN_MNIST/MNIST.png" | prepend: site.baseurl }}">


## Goal of this implementation

Our aim should be to implement a simple generative network based on GANs to train on MNIST dataset and then generate the images. 



## Implementation

### Skeleton for GAN

#### Lets start with the main function

we should create structure of how our high level pipeline should look like:

```python
def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = # TODO

    # Build model.
    model = Gan() # TODO

    # Start training
    train() # TODO
    
    # Generate samples
    generate() # TODO


if __name__ == "__main__":
    tf.app.run()
```

So this will be our `main.py` file. 

#### Now lets move on to getting data

We are lucky that `tensorflow` provides the data in tensor format and we just have to use the following lines of code. 



```python 
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
```

we just literally need to paste it one of the file and lets call it `input_data.py`. Getting data is done !

#### Update the main function

```python
import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan() # TODO

    # Start training
    train() # TODO
    
    # Generate samples
    generate() # TODO


if __name__ == "__main__":
    tf.app.run()
```

#### Lets create the GAN class

lets create a `constructor` (what we do usually !) and create a `generator` function and a `discriminator` function. We might also need two more functions for `discriminator loss` and `generator loss` . Lets create one function for `generating samples` as well (Not sure if we need it or not!)

```python
class Gan(object):
    """Adversary based generator network.
    """

    def __init__(self):
        """Initializes a GAN"""
		pass

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network"""
        
		y = None
        return y

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator."""

        l = None
        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image."""
			x_hat = None
            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator."""
        
        l = None
        return l

    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np."""
        out = None
        return out

```

save this in a folder `models/gan.py` 

Now we have to start filling all the functions that we have created inside the class `Gan`. We will start with constructor of the class. We will pass 2 variables in constructor `ndims` and `nlatent` to initialize our model. We need to build the graph for the model here. For that, we need 2 placeholders as well for the image and the latent variable. Additionally, we need to assign discriminator and generator loss here as well. Then, most importantly, we need to create separate variables for both generator and discriminator network. This will make sure the optimizer work independently on both of the networks. Finally, we have to initialize the tensorflow session here. 

```python
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Learning rates
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        # self.g_learning_rate_placeholder = tf.placeholder(tf.float32)

        # Add optimizers for appropriate variables
        d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='d_optimizer').minimize(self.d_loss, var_list=d_var)

        g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='g_optimizer').minimize(self.g_loss, var_list=g_var)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
```



### Building Model

#### Generator Network

We have a total of 2 functions for generator block : `_generator` and `_generator_loss`. We have to make sure that we use correct set of variables (i.e. the variables that are assigned to the generator network) in the generator block. Therefore, using `scope` is important here. We will just use one dense layer as hidden layer which takes latent variable `z` as the input and maps to 64 neurons. We will use `ReLU` for the activation function. We will now take 64 neurons as input and spits out an image with another dense layer with `sigmoid` activation function.  

```python
def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:

            # Input layer
            hidden_1 = tf.layers.dense(
                inputs=z, units=64, activation=tf.nn.relu, name='inputs-layer', reuse=reuse)

            x_hat = tf.layers.dense(
                inputs=hidden_1, units=self._ndims, activation=tf.nn.sigmoid)
            return x_hat
```

Now, we need to complete the loss function. We will be using the `cross_entropy` loss. This loss function takes arguments as the probability score given by discriminator as logits and constant value of 1. This is because the generator wants to optimize its weights such that the discriminator always produce 1 for for the image produced by the generator. Note, we have to use `sigmoid_cross_entropy_with_logits` to make sure that discriminator network always give probability which lies between 0 and 1. 

```python 
    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_hat), logits=y_hat))
        return l
```



### Discriminator Network

After dealing with generator, discriminator's network looks to be similar but simpler. Again, we will only use one hidden layer of 512 neurons with `ReLU` as activation function. Here we don't need to make sure that discriminator spits only probability because we have already made this sure in our generator loss function by using sigmoid function with cross entropy.  

```python
    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:

            # Input
            hidden_1 = tf.layers.dense(
                inputs=x, units=512, activation=tf.nn.relu, reuse=reuse)

            y = tf.layers.dense(
                inputs=hidden_1, units=1, activation=None)
            return y
```

Discriminator's job is to optimize its parameters such that it assign high probability to ground truth images and low probability to the generated images by the generator network. We will again use the `sigmoid_cross_entropy_with_logits` for the ground truth loss and the generated loss. 

```python
    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        gt_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y), logits=y, name="d_loss_gt")
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_hat), logits=y_hat, name="d_loss_gen")
        total_loss = gt_loss + gen_loss
        l = tf.reduce_mean(total_loss)
        return l
```



#### Additional helper functions

We have one more function to go! this function will generate sample images from points in the latent space. The session evaluation is quite straightforward :) 

```python
    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        out = self.x_hat.eval(session=self.session, feed_dict={self.z_placeholder: z_np})
        return out
```



### Training stuff

The model training is done in the main function. We have to specify some hyperparameters like learning rate, batch size and number of iterations here. According to original [GAN article by Ian Goodfellow](https://arxiv.org/abs/1406.2661), it is mentioned that discriminator might need $k$ times more iteration as compared to generator for competitive training. Fortunately for us, we are able to perform training with just one iteration each for both networks. For training, we are sampling from uniform distribution with latent space of 10 dimensions. Finally, we will save the image generated by the generator after completion of the training. Hence, we don't need a separate testing loop. 

```python
def train(model, mnist_dataset, learning_rate=0.001, batch_size=16,
          num_steps=50000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    # Iterations for generator
    g_iters = 1

    # Iterations for discriminator
    d_iters = 1

    print('Batch Size: %d, Total epoch: %d, Learning Rate : %f' %
          (batch_size, num_steps, learning_rate))

    print('Start training ...')

    for epoch in range(0, num_steps):

        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, 10])
        
        # Train generator and discriminator
        for train_discriminator in range(d_iters):
            _, d_loss = model.session.run(
                [model.d_optimizer, model.d_loss],
                feed_dict={model.x_placeholder: batch_x,
                           model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

        batch_z = np.random.uniform(-1, 1, [batch_size, 10])
        for train_generator in range(g_iters):
            _, g_loss = model.session.run(
                [model.g_optimizer, model.g_loss],
                feed_dict={model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )
            
	# Saving Image
    std = 1
    x_z = np.linspace(-3 * std, 3 * std, 20)
    y_z = np.linspace(-3 * std, 3 * std, 20)

    out = np.empty((28 * 20, 28 * 20))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.random.uniform(-1, 1, [16, 10])
            img = model.generate_samples(z_mu)
            out[x_idx * 28:(x_idx + 1) * 28,
                y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
            plt.imsave(path, out, cmap="gray")
```



Now, we are done with implementation. The most updated code from the repository can be found here. 



## Results

After a full 50000 epochs, we are glad to present you our results. 

<img src= "{{ "/img/Naman/GAN_MNIST/Result.gif" | prepend: site.baseurl }}">

The final result is given below:

<img src= "{{ "/img/Naman/GAN_MNIST/final.png" | prepend: site.baseurl }}">


## Analysis

We can clearly observe the indication of `mode collapse` in our model as the most of the digits are dominated by either 9, 7, 1 or 4. Honestly, this is the best result you could expect from just one hidden layer. This could be because of various reasons like low model complexity, weak discriminator and even because of our loss function. Overall, we have achieved our aim of writing a fairly simple generative network that performs fairly well. 



