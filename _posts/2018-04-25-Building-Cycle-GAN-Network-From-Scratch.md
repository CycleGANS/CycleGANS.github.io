---
layout:     notebook
title:      Building Cycle GAN Network From Scratch
author:     Naman Shukla
tags:       GANS CycleGANs PatchGANs Discriminator Generator DeepNets Vision
subtitle:   Detailed implementation for building the network components
category:  project1
---

<img src= "{{ "/img/Naman/Code/code.gif" | prepend: site.baseurl }}">


## Quick Recap !

Up till now, we have successfully build our `Generator` and `Discriminator` network. Now it time to integrate this into a single model for cycle consistent network or `Cycle GAN`.  To achieve that, here's the game plan : First finish the data handling which involves all preprocessing of the data. Then, we have to implement the training and test for the network. Finally, integrate into one single module.  

> **NOTE**: This blog is going to be pretty heavy implementation oriented post to be honest! we recommend to make yourself familiar with the previous posts on cycle GANs first if necessary. 

## Data Handling 

We will be testing our implementation on standard dataset for unpaired image to image translation is available at [EECS UC Berkeley’s CycleGAN](http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) web page. For preprocessing, we will be using the following code. This code will crop, resize and convert it into proper batch tensor that can be used for training and testing process. 

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



## Network Training

This is the most crucial part of the entire implementation. The entire code is available in our [repository](https://github.com/CycleGANS/V1.0/blob/master/CycleGAN/train.py).  We will be following the following algorithm to create the training method. 

> We will be following the same network flow diagram presented in the theoretical blog. Its a good idea to keep this flow chart in the mind :)

First we will create a place holder for the the images. 

```python
# Creating placeholder for images
X = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
Y = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
GofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
FofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
```

Then we will initialize the Generator and Discriminator networks

```python
# Creating the generators and discriminator networks
GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
GofFofY = gen.generator(FofY, no_of_residual_blocks, scope='G', output_channels=64)
FofGofX = gen.generator(GofX, no_of_residual_blocks, scope='F', output_channels=64)

D_Xlogits = dis.build_gen_discriminator(X, scope='DX')
D_FofYlogits = dis.build_gen_discriminator(FofY, scope='DX')
D_Ylogits = dis.build_gen_discriminator(Y, scope='DY')
D_GofXlogits = dis.build_gen_discriminator(GofX, scope='DY')
```

Now we have to calculate all of our losses : Adversarial and Cyclic. 

```python
# Adversary and Cycle Losses for G
G_adv_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.ones_like(D_GofXlogits)))
G_cyc_loss = tf.reduce_mean(tf.abs(GofFofY - Y)) * G_cyc_loss_lambda        # Put lambda for G cyclic loss here
G_tot_loss = G_adv_loss + G_cyc_loss

# Adversary and Cycle Losses for F
F_adv_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.ones_like(D_FofYlogits)))
F_cyc_loss = tf.reduce_mean(tf.abs(FofGofX - X)) * F_cyc_loss_lambda        # Put lambda for F cyclic loss here
F_tot_loss = F_adv_loss + F_cyc_loss

# Total Losses for G and F
GF_tot_loss = G_tot_loss + F_tot_loss

# Losses for DX
DX_real_loss = tf.reduce_mean(tf.squared_difference(D_Xlogits, tf.ones_like(D_Xlogits)))
DX_fake_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.zeros_like(D_FofYlogits)))
DX_tot_loss = (DX_real_loss + DX_fake_loss) / 2

# Losses for DY
DY_real_loss = tf.reduce_mean(tf.squared_difference(D_Ylogits, tf.ones_like(D_Ylogits)))
DY_fake_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.zeros_like(D_GofXlogits)))
DY_tot_loss = (DY_real_loss + DY_fake_loss) / 2
```

Now, its time to optimize the variables from each of the networks.

```python
# Optimization
# Getting all the variables that belong to the different networks
# I.e. The weights and biases in G, F, DX and DY
# This gets all the variables that will be initialized
network_variables = tf.trainable_variables()  
GF_variables = [variables for variables in network_variables if 'G' in variables.name or 'F' in variables.name]
DX_variables = [variables for variables in network_variables if 'DX' in variables.name]
DY_variables = [variables for variables in network_variables if 'DY' in variables.name]

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5) 
GF_train_step = optimizer.minimize(GF_tot_loss, var_list=GF_variables)
DX_train_step = optimizer.minimize(DX_tot_loss, var_list=DX_variables)
DY_train_step = optimizer.minimize(DY_tot_loss, var_list=DY_variables)
```

We have to load the data now, we will be using the data handler functions that we have created earlier. 

```python
# Session on GPU
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Obtaining dataset
# Training data
""" Need to define getdata"""
# dataset = 'horse2zebra'
Xpath = glob.glob('./Datasets/' + dataset + '/trainA/*.jpg')
Ypath = glob.glob('./Datasets/' + dataset + '/trainB/*.jpg')
X_data = io.getdata(sess, Xpath, batch_size)     # Need to define getdata
Y_data = io.getdata(sess, Ypath, batch_size)

# Test data
X_test_path = glob.glob('./Datasets/' + dataset + '/testA/*.jpg')
Y_test_path = glob.glob('./Datasets/' + dataset + '/testB/*.jpg')
X_test_data = io.getdata(sess, X_test_path, batch_size)     # Need to define getdata
Y_test_data = io.getdata(sess, Y_test_path, batch_size)     # Need to define getdata
```



Finally, its time to write the training loop !! 

```python
# Training
no_of_iterations = 0
for i in range(1, epochs + 1):
    for j in range(1, no_of_batches + 1):
        no_of_iterations += 1

        # Define Batch
        X_batch = io.batch(sess, X_data)
        Y_batch = io.batch(sess, Y_data)

        # Creating fake images for the discriminators
        GofXforDis, FofYforDis = sess.run([GofX, FofY], feed_dict={X: X_batch, Y: Y_batch})

        DX_output = sess.run([DX_train_step], feed_dict={X: X_batch, FofY: FofYforDis})

        DY_output = sess.run([DY_train_step], feed_dict={Y: Y_batch, GofX: GofXforDis})

        GF_output = sess.run([GF_train_step], feed_dict={X: X_batch, Y: Y_batch})

        # To see what some of the test images look like after certain number of iterations
        if no_of_iterations % 400 == 0:
            X_test_batch = io.batch(sess, X_test_data)  # Define batch
            Y_test_batch = io.batch(sess, Y_test_data)

            [GofX_sample, FofY_sample, GofFofY_sample, FofGofX_sample] = sess.run([GofX, FofY, GofFofY, FofGofX], feed_dict={X: X_test_batch, Y: Y_test_batch})

            # Saving sample test images
            for l in range(batch_size):

                new_im_X = np.zeros((image_shape, image_shape * 3, 3))
                new_im_X[:, :image_shape, :] = np.asarray(X_test_batch[l])
                new_im_X[:, image_shape:image_shape * 2, :] = np.asarray(GofX_sample[l])
                new_im_X[:, image_shape * 2:image_shape * 3, :] = np.asarray(FofGofX_sample[l])

                new_im_Y = np.zeros((image_shape, image_shape * 3, 3))
                new_im_Y[:, :image_shape, :] = np.asarray(Y_test_batch[l])
                new_im_Y[:, image_shape:image_shape * 2, :] = np.asarray(FofY_sample[l])
                new_im_Y[:, image_shape * 2:image_shape * 3, :] = np.asarray(GofFofY_sample[l])

                scipy.misc.imsave('./Output/Train/' + dataset + '/X' + str(l) + '_Epoch_(%d)_(%dof%d).png' % (i, j, no_of_batches), _to_range(new_im_X, 0, 255, np.uint8))
                scipy.misc.imsave('./Output/Train/' + dataset + '/Y' + str(l) + '_Epoch_(%d)_(%dof%d).png' % (i, j, no_of_batches), _to_range(new_im_Y, 0, 255, np.uint8))

                print("Epoch: (%3d) Batch Number: (%5d/%5d)" % (i, j, no_of_batches))

sess.close()

```

> **Additional Helper Function**: This function makes sure that the range of the images generated is between 0 and 255. This function is taken [LynnHo's repository](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/blob/master/image_utils.py). 
>
> ```pytho
> def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
>     # transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
>     assert \
>         np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
>         and (images.dtype == np.float32 or images.dtype == np.float64), \
>         'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
>     if dtype is None:
>         dtype = images.dtype
>     return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)
> ```



## Network Testing

After some heavy lifting, we deserve something to chill. This is it! Here we just have to implement few steps from training but with slightly different structure.  

```python
def test(dataset_str='horse2zebra', img_width=256, img_height=256):
    """Test and save output images.
    Args:
        dataset_str: Name of the dataset
        X_path, Y_path: Path to data in class X or Y
    """
    image_shape = img_width

    if image_shape == 256:
        no_of_residual_blocks = 9
    elif image_shape == 128:
        no_of_residual_blocks = 6

    # Session on GPU
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # X and Y are for real images.
        X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        Y = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])

        # Build graph for generator to produce images from real data.
        GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
        FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
        # Convert transformed images back to original one (cyclic).
        Fof_GofX = gen.generator(GofX, no_of_residual_blocks, scope='F', output_channels=64)
        Gof_FofY = gen.generator(FofY, no_of_residual_blocks, scope='G', output_channels=64)

        saver = tf.train.Saver(None)

        # Restore checkpoint.
        # --------------- Need to implement utils!!!!! ----------------
        try:
            saver.restore(sess, tf.train.latest_checkpoint("./Checkpoints/" + dataset_str))
            print('Checkpoints Restored !')
        except:
            raise Exception('No checkpoint available!')

        # Load data and preprocess (resize and crop).
        X_path_ls = glob.glob('./Datasets/' + dataset_str + '/testA/*.jpg')
        Y_path_ls = glob.glob('./Datasets/' + dataset_str + '/testB/*.jpg')

        batch_size_X = len(X_path_ls)
        batch_size_Y = len(Y_path_ls)

        X_data = getdata(sess, X_path_ls, batch_size_X)
        Y_data = getdata(sess, Y_path_ls, batch_size_Y)

        # Get data into [batch_size, img_width, img_height, channels]
        X_batch = batch(sess, X_data)
        Y_batch = batch(sess, Y_data)

        print('test data :' + dataset_str + '- uploaded!')
        # Feed into test procedure to test and save results.
        X_save_dir = './Output/Test/' + dataset_str + '/testA'
        Y_save_dir = './Output/Test/' + dataset_str + '/testB'

        _test_procedure(X_batch, sess, GofX, Fof_GofX, X, X_save_dir, image_shape)
        _test_procedure(Y_batch, sess, FofY, Gof_FofY, Y, Y_save_dir, image_shape)
```

```python
def _test_procedure(batch, sess, gen_real, gen_cyc, real_placeholder, save_dir, image_shape):
    """Procedure to perform test on a batch of real images and save outputs.
    Args:
        gen_real: Generator that maps real data to fake image.
        gen_cyc: Generator that maps fake image back to original image.
        real_placeholder: Placeholder for real image.
        save_dir: Directory to save output image.
    """
    print('Test Images sent to generator..')
    gen_real_out, gen_cyc_out = sess.run([gen_real, gen_cyc],
                                         feed_dict={real_placeholder: batch})
    print('Images obtatined back generator..')
    for i in range(batch.shape[0]):
        # A single real image in batch.
        real_img = batch[i]
        
        new_im = np.zeros((image_shape, image_shape * 3, 3))
        new_im[:, :image_shape, :] = np.asarray(real_img)
        new_im[:, image_shape:image_shape * 2, :] = np.asarray(gen_real_out[i])
        new_im[:, image_shape * 2:image_shape * 3, :] = np.asarray(gen_cyc_out[i])

        scipy.misc.imsave(save_dir + 'Image(%d).png' % (i), _to_range(new_im, 0, 255, np.uint8))
        print("Save image.")
```

