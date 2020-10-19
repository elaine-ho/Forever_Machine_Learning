---
layout: default
title:  Proposal
---

# Proposal

### Summary

This project’s objective is to remove rain from photos of scenes in Minecraft. We will use machine learning algorithms to transform a Minecraft scene image taken under rainy conditions (input) into an image of the same scene without rain (output). Our dataset will be a self-made collection of paired images depicting a scene under both rainy and normal conditions.This project will allow users to share clear gameplay even in rainy weather, and will show the effectiveness of rain removal using generative adversarial networks in a simulated realistic environment of Minecraft.

### Algorithms

We will build generative adversarial networks (GANs) models that utilize various convolutional and dense layers.

### Evaluation Plan

The base case for our project will be rain against a brick wall. Hence, the algorithm will be tasked with just removing the rain and not adjusting the sky and lighting. Afterwards, we will tackle simple shaped landscapes with no objects. Eventually, we will convert images involving buildings and scenery. 

#### Quantitative 

Once a de-rained image has been produced, its pixels will be compared to the same scene in clear weather from our dataset of paired images. Our goal is for at least 70% of the pixels in the output image to be accurately matching their corresponding pixels in the clear image. We can verify the internals of the algorithm work by pixel accuracy improving over time. We will build multiple baseline models using simple GANs and increase complexity of the models according to existing rain removal GANs algorithms that have been published.

#### Qualitative 

Human participants can be given a mixed set of de-rained images and originally clear images and asked if the image has been altered or processed in some way. If the image was de-rained, the participants can describe how accurately they think the produced image fares compared to clear paired image. They could describe which attributes make it look fabricated, giving an insight to what is regarded as the Minecraft aesthetic. It would be impressive to produce results with video or live footage as well.

### Appointment with Instructor

4:30 PM on Thursday, October 22
