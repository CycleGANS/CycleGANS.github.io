---
layout:     notebook
title:      Results and Analysis on Cycle GAN Implementation
author:     Naman Shukla
tags:       GANS CycleGANs PatchGANs Discriminator Generator DeepNets Vision
subtitle:   A discussion about the implemented results and outcomes
category:  project1
---

## Check lists for Cycle GAN

We have come a long way from where we've started. Lets see what we have done till now and also watch out for upcoming stuff!

- Understanding & Implementing - Convolutional Neural Networks. `DONE`
- Understanding & Implementing - Simplified Generative Adversarial Networks. `DONE`
- Theory behind cycle consistent image 2 image translation with GANs. `DONE`
- Implementing Cycle GAN from scratch. `DONE`
- Analyzing different datasets with our network. 

It's time to test our implementation on slandered datasets and analyze the performance of the network. 



## Implementation

If you want to implement our code off the shelf, you can find the entire code for `Cycle GAN` network in our [repository](https://github.com/CycleGANS/V1.0/tree/master/CycleGAN). To download specific datasets, please refer to the `Resources` section of the [About Page](https://cyclegans.github.io/about/). 

#### Dependencies

> **Note** : The following packages must be installed in your machine if you want to run CycleGAN : 
>
> 1. glob
> 2. Tensorflow
> 3. scipy
> 4. numpy
> 5. Pillow

The dependencies can be installed by following the commands :

```
git clone https://github.com/CycleGANS/V1.0.git
cd CycleGAN
sh ./download_dataset.sh horse2zebra
mv datasets Datasets
pip install -r requirements.txt
```



#### Running the code

> **Note** : This is a heavy code to execute on a CPU. A GPU is highly recommended. We have used Blue Waters - K80 Graphical Processing Unit for this project.  You can also download our trained weights available [here](https://uofi.box.com/s/w3o6gnic0uxrrxgo3ugz2f2vq2dj0gev) and keep it in the checkpoints directory. 

You can run our code by following command:

```
python3 main.py
```



## Results

<img src= "{{ "/img/Naman/Results/top_best.png" | prepend: site.baseurl }}">

We have tested our network implementation in 5 different datasets : `horse2zebra`, `apple2orange`, `Summer2winter`, `monet2photo` and `vangogh2photo` . Here are some of the results on each of the following :



#### Horse - Zebra

<img src= "{{ "/img/Naman/Results/horsetozebra_good.png" | prepend: site.baseurl }}">

<center><em>Figure 1: Horse to Zebra translation</em></center>

<img src= "{{ "/img/Naman/Results/zebratohorse_good.png" | prepend: site.baseurl }}">

<center><em>Figure 2: Zebra to Horse translation</em></center>

