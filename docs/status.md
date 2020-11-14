---
Layout: default
Title: Status
---

# Status Report

<iframe width="560" height="315" src="https://www.youtube.com/embed/GK7XkF3Pivk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Project Summary

This project’s objective is to generate minecraft scenes in different environment conditions. Using machine learning algorithms, we explore supervised learning tasks of converting screenshots of Minecraft between weather conditions and times of day. Our initial objective is to build a model that converts a rainy Minecraft landscape image into a clear Minecraft image of the same landscape without rain. Our dataset is a self-made collection of pairs of images depicting a given scene in two different conditions, rainy and clear weather. We also have data of snow and desert areas with no rain, as these regions do not have those properties. All our current data is taken at noon in Minecraft time. Going forward, we will explore the task of converting an image of a scenery between different times of day, accounting for sky color and lighting changes.This project will allow users to share gameplay in different environment conditions and showcase the effectiveness of machine learning algorithms in detecting and converting weather effects in a simulated realistic environment of Minecraft.

### Approach

To convert Minecraft landscapes from rain to sunny and vice versa, we took inspiration from image to image translation GANs models. Since the combination of Malmo and Minecraft allowed us to generate paired data, we decided to use only the generator to create a deep neural network with a U-Net architecture. This encoder and decoder structure of the U-net works to downsample and upsample the image with skip connections between mirrored layers as shown below from Phillip Isola’s paper. The layers are comprised of convolutions, rectifiers, and batch normalization. The encoder down samples the image to 1x1 through 8 blocks down and up samples back to 256x256 with 8 blocks up, gradually reducing the filter size as recommended in the referenced image to image translation paper.

![Encoder-Decoder Unets](/images/unets.png)

Once the layers are constructed, other parameters are set including the optimizer using Adam with a learning rate of 0.0001 and beta values of 0.5 and 0.999. The L1 loss function multiplied by a weight of 100 concatenated to the binary cross entropy loss is used (both equations listed below). Both loss functions are used because the BCE loss helps to smooth the images as it has a bias towards 0.5 while the L1 loss is used to maintain the original rgb of the input image.

![Binary Cross-Entropy / Log Loss](/images/log-loss.png)

![Loss Function](/images/l1-loss-function.png)

### Evaluation

As a first step for evaluating our images, we checked the loss functions to make sure it is steadily decreasing. The binary cross entropy loss was not going down as much but we were able to see a drop in the L1 loss.

![loss plots](/images/lossPlots_10_10.png)

The second form of evaluation was through visually comparing side by side photos of the input, predicted, and target data. As seen below, the Minecraft rain images to sunny performed fairly well in removing the blue streaks. The images were slightly blurred because of the smoothing that occurs to remove the rain. The sunny to rain conversion was slightly more difficult for the model through visual inspection as the images were darkened for the rain effect but was not able to portray blue streaks.

Generated images for removing rain: 

![de-rain image 1](/images/derain1.png)

![de-rain image 2](/images/derain2.png)

![de-rain image 3](/images/derain4.png)

![de-rain image 4](/images/derain5.png)

![de-rain image 5](/images/derain6.png)

![de-rain image 6](/images/derain7.png)

Inverse mapping of our model to add rain: 

![de-rain image 1](/images/rain1.png)

![de-rain image 2](/images/rain2.png)

![de-rain image 3](/images/rain3.png)

![de-rain image 4](/images/rain4.png)

### Remaining Goals and Challenges

In the remaining weeks of the quarter we will work to improve the performance of the model on paired rainy and clear weather images. This can be improved by training the model for extended periods of time and possibly tweaking the Unet layers. Our reach goal is to build a similar model but for night and day in Minecraft rather than weather.

In terms of evaluation, we want to try the qualitative evaluation approach described in our proposal, of having human participants explain what differences they see between the outputted images and the true pair images. Although we have stated our judgement of the produced images above, we still need human participants to rate the quality of our image conversion.


### Resources Used

Godoy, D. (2019, February 07). Understanding binary cross-entropy / log loss: A visual explanation. Retrieved November 14, 2020, from https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

Image-to-Image Translation with Conditional Adversarial Networks.
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017.

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
Jun-Yan Zhu*, Taesung Park*, Phillip Isola, Alexei A. Efros. In ICCV 2017. (* equal contributions)

What Are L1 and L2 Loss Functions? (n.d.). Retrieved November 14, 2020, from https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions
