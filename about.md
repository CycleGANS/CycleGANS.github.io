---
layout: page
title: "About"
description: "Cycle GAN - Computer Vision @ UIUC "
header-img: "img/home-bg.jpg"
---

### Description: 

In this project, the idea is to implement the algorithm developed by `Jun-Yan Zhu`, `Taesung Park`, `Phillip Isola` and `Alexei A. Efros` in their paper “[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)”. 

​    <center><img src= "{{ "/img/utils/paper.jpg " | prepend: site.baseurl }}" style="width: 60%; margin-left: 1%; margin-right: 1%;"></center>

This algorithm learns a mapping function  $G: X \mapsto Y$ using an adversarial loss such that $G(X)$ cannot be distinguished from $Y$, where $X$ and $Y$ are the input and output images respectively. The algorithm also learns an inverse mapping function $F: Y \mapsto X$ using a cycle consistency loss such that $F(G(X))$ is indistinguishable from $X$. However, the major aspect in which this implementation truly shines is that it does not require the $X$ and $Y$ pairs to exist, i.e. image pairs are not needed to train this model. This is highly beneficial as such pairs are not necessarily always available or tend to be expensive monetarily. An application of this could be used in movies, where, if a movie crew was unable to shoot a scene at a particular location during the summer season and it is now winter, the movie crew can now shoot the scene and use this algorithm to generate scenes which look like they were shot during the summer. Other areas in which this algorithm can be applied include image enhancement, image generation from sketches or paintings, object  transfiguration, etc. The algorithm proves to be superior to several prior methods.



### Goal: 



##### Minimum Goal: 
Implement the method proposed in the paper and test it on test and self provided images. Eg. apples to oranges and vice versa, horses to zebras and vice versa, etc.

##### Maximum Goal: 
Implement this algorithm and propose certain modifications to enhance the outputs and also make it work on videos. Eg. the seasonal change video discussed in the introduction.



### Reservations: 

Although, team members have some basic understanding of neural networks, understanding the underlying network graph for unpaired image to image translation could be challenging. Training the entire dataset could be a mammoth task both computationally and schematically, hence implementing a toy example for unpaired image to image translation is the minimum task for this project. 



### Relationship to your background: 

All three team members are graduate (Masters’) students in the Department of Industrial Engineering with a concentration in Advanced Analytics. Ziyu and Naman are currently enrolled in the CS446: Machine Learning course which requires them to implement various machine learning algorithms on TensorFlow. The course will also be providing them with an introduction to Generative Adversarial Networks (GANs). Harshad has worked on research projects with his advisor that required him to work with Machine Learning algorithms and implement small Neural Networks on Keras. All three members want to advance their knowledge in Deep Learning and Computer Vision as they desire to pursue their careers as Machine Learning Engineers/Data Scientists. The team does not have prior experience in computer vision and deep learning research. This project will provide them with an excellent opportunity to learn about neural networks and GAN’s. 



### Resources: 

The standard dataset for unpaired image to image translation is available at [EECS UC Berkeley’s CycleGAN](http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) web page. 

This data set have following image clusters :

|          Data          |       Date       | Size |
| :--------------------: | :--------------: | :--: |
|     ae_photos.zip      | 2017-04-03 22:06 | 10M  |
|    apple2orange.zip    | 2017-03-28 13:51 | 75M  |
|   cezanne2photo.zip    | 2017-03-28 13:51 | 267M |
|     cityscapes.zip     | 2017-03-29 03:22 | 267M |
|      facades.zip       | 2017-03-29 23:23 | 34M  |
|    horse2zebra.zip     | 2017-03-28 13:51 | 111M |
| iphone2dslr_flower.zip | 2017-03-30 12:05 | 324M |
|        maps.zip        | 2017-03-26 19:17 | 1.4G |
|    monet2photo.zip     | 2017-03-26 19:17 | 291M |
|   Summer2winter.zip    | 2017-03-26 19:17 | 126M |
|    ukiyoe2photo.zip    | 2017-03-26 19:17 | 279M |
|   vangogh2photo.zip    | 2017-03-26 19:17 | 292M |

> **Note** : The script to download the dataset can be found at [Cycle GAN repository](https://github.com/CycleGANS/CS543CycleGANsProject) and can we used by following command :
>
> ```
> sh ./download_dataset.sh <name of dataset>
> ```

The team requires access to BlueWaters for training the network as none of the team members have a GPU or access to any cloud computing platforms. 