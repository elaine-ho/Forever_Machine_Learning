---
layout: default
title:  Proposal
---

# Proposal

### Summary

This project’s objective is to remove rain from photos of scenes in Minecraft. We will use machine learning algorithms to transform a Minecraft scene image taken under rainy conditions(input) into a clearer image of the same scene without rain effects. (output). Our dataset will be a self-made collection of paired images depicting a scene under both rainy and normal conditions.This project would allow users to share clearer images of scenes taken under rainy conditions and could possibly be applied to images of scenes in other video games as well.

### Algorithms

We will be using generative adversarial networks (GAN) and potentially convolutional neural networks (CNN) as well.

### Evaluation Plan

The base case for our project will be rain against a brick wall. Hence, the algorithm will be tasked with just removing the rain and not adjusting the sky and lighting. Afterwards, we will tackle simple shaped landscapes with no objects. Eventually, we will convert images involving buildings and scenery. 

#### Quantitative 

Once a de-rained image has been produced, its pixels will be compared to the same scene in clear weather from our dataset of paired images. Our goal is for at least 70% of the pixels in the output image to be accurately matching their corresponding pixels in the clear image. We can verify the internals of the algorithm work by pixel accuracy improving over time. 

#### Qualitative 

Human participants can be given a mixed set of de-rained images and originally clear images and asked if the image has been altered or processed in some way. If the image was de-rained, the participants can describe how accurately they think the produced image fares compared to clear paired image. They could describe which attributes make it look fabricated, giving an insight to what is regarded as the Minecraft aesthetic. It would be impressive to produce results with video or live footage as well.

### Appointment with Instructor

4:30 PM on Thursday, October 22
