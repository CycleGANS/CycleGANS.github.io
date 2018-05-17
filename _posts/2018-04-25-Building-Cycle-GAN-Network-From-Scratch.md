---
layout:     notebook
title:      Building Cycle GAN Network From Scratch
author:     Naman Shukla
tags:       GANS CycleGANs PatchGANs Discriminator Generator DeepNets Vision
subtitle:   Detailed implementation for building the network components
category:  project1
---

<img src= "{{ "/img/Naman/Code/code.gif" | prepend: site.baseurl }}">


## Generator

The generator function have 4 high level operations : `Batch Normalization`, `Convolution`, `De-Convolution` and `Residual Block` . Lets follow the same order and write the generator network. 

### Batch Normalization 

This function is relatively straightforward. Given a set of `Logits` we have to normalize them according to $BN(x_i) = \gamma(\frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}) + \beta$.  We will add small float added to variance to avoid dividing by zero. 

```python
# Function for Batch Normalization
def batchnorm(Ylogits):
    bn = tf.contrib.layers.batch_norm(Ylogits, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
    return bn
```



### Convolution & De-Convolution



```python
# Function for Convolution Layer
def convolution_layer(input_images, filter_size, stride, o_c=64, padding="VALID", scope_name="convolution"):
    with tf.variable_scope(scope_name):
        conv = tf.contrib.layers.conv2d(input_images, o_c, filter_size, stride, padding=padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return conv
```

```python
# Function for deconvolution layer
def deconvolution_layer(input_images, o_c, filter_size, stride, padding="VALID", scope_name="deconvolution"):
    with tf.variable_scope(scope_name):
        deconv = tf.contrib.layers.conv2d_transpose(input_images, o_c, filter_size, stride, activation_fn=None,
                                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return deconv
```



```python
# Function for Residual Block
def residual_block(Y, scope_name="residual_block"):
    with tf.variable_scope(scope_name):
        Y_in = tf.pad(Y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        Y_res1 = tf.nn.relu(batchnorm(convolution_layer(Y_in, filter_size=3, stride=1, o_c=output_channels * 4, scope_name="C1")))
        Y_res1 = tf.pad(Y_res1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        Y_res2 = batchnorm(convolution_layer(Y_res1, filter_size=3, stride=1, padding="VALID", o_c=output_channels * 4, scope_name="C2"))

        return Y_res2 + Y
```



```python
with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    # Need to pad the images first to get same sized image after first convolution
    input_imgs = tf.pad(input_imgs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

    YD0 = tf.nn.relu(batchnorm(convolution_layer(input_imgs, filter_size=7, stride=1, o_c=output_channels, scope_name="D1")))
    YD1 = tf.nn.relu(batchnorm(convolution_layer(YD0, filter_size=3, stride=2, o_c=output_channels * 2, padding="SAME", scope_name="D2")))
    YD2 = tf.nn.relu(batchnorm(convolution_layer(YD1, filter_size=3, stride=2, o_c=output_channels * 4, padding="SAME", scope_name="D3")))

    # For Residual Blocks
    for i in range(1, no_of_residual_blocks + 1):
        Y_res = residual_block(YD2, scope_name="R" + str(i))

        # For Upsampling
        YU1 = tf.nn.relu(batchnorm(deconvolution_layer(Y_res, output_channels * 2, filter_size=3, stride=2, padding="SAME", scope_name="U1")))
        YU2 = tf.nn.relu(batchnorm(deconvolution_layer(YU1, output_channels, filter_size=3, stride=2, padding="SAME", scope_name="U2")))
        Y_out = tf.nn.tanh(convolution_layer(YU2, filter_size=7, stride=1, o_c=3, padding="SAME", scope_name="U3"))

        return Y_out
```



## Discriminator



```python
def _conv2d_layer(input_conv, num_filter=64, filter_h=4, filter_w=4, stride_h=1, stride_w=1, stddev=0.02,
                  padding="VALID", name="conv2d", do_norm=True, do_relu=True, relu_alpha=0):
    """Convolution layer for discriminator.
    Supports normalization for image instance and leaky ReLU.

    Note:
        relu_alpha: Slope when x < 0, used in max(x, alpha*x).
    """

    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(input_conv, num_filter, filter_h, stride_h, padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = _normalization(conv)

        if do_relu:
            if(relu_alpha == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = _leaky_relu(conv, relu_alpha, "leaky_relu")

        return conv
```



```python
def _normalization(x):
    """Adapted from hardikbansal's code. Will change it later."""
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out
```



```python
def _leaky_relu(x, relu_alpha, name="leaky_relu"):
    with tf.variable_scope(name):
        return tf.maximum(x, relu_alpha * x)
```



```python
with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    filter_size = 4

    layer1 = _conv2d_layer(input_images, num_filters, filter_size, filter_size, 2, 2, 0.02,
                           "SAME", "conv1", do_norm=False, do_relu=True, relu_alpha=0.2)
    layer2 = _conv2d_layer(layer1, num_filters * 2, filter_size, filter_size, 2, 2, 0.02,
                           "SAME", "conv2", do_norm=True, do_relu=True, relu_alpha=0.2)
    layer3 = _conv2d_layer(layer2, num_filters * 4, filter_size, filter_size, 2, 2, 0.02,
                           "SAME", "conv3", do_norm=True, do_relu=True, relu_alpha=0.2)
    layer4 = _conv2d_layer(layer3, num_filters * 8, filter_size, filter_size, 1, 1, 0.02,
                           "SAME", "conv4", do_norm=True, do_relu=True, relu_alpha=0.2)
    layer5 = _conv2d_layer(layer4, 1, filter_size, filter_size, 1, 1, 0.02,
                           "SAME", "conv5", do_norm=False, do_relu=False)

    return layer5
```



## Data Handling 



```python
def _parse_image(path):

    load_size = 286
    crop_size = 256
    img = tf.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [load_size, load_size])
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    img = tf.random_crop(img, [crop_size, crop_size, 3])
    img = img * 2 - 1
    return img


def getdata(sess, paths, batch_size, shuffle=True):

    prefetch_batch = 2
    num_threads = 16
    buffer_size = 4096
    repeat = -1

    _img_num = len(paths)

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    # The map method takes a map_func argument that describes how each item in the Dataset should be transformed.
    dataset = dataset.map(_parse_image, num_parallel_calls=num_threads)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # this transformation combines consecutive elements of this dataset into batches.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Repeats this dataset count times | repeated indefinitely if -1
    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset.make_one_shot_iterator().get_next()


def batch(sess, dataset):

    return sess.run(dataset)

```

