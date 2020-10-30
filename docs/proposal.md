---
layout: default
title:  Proposal
---

# Proposal

### Summary

This project’s objective is to generate realistics minecraft scenery in different environment conditions. Using machine learning algorithms, we will explore supervised learning tasks of converting between various weather conditions, day and night, and if time permits, multiple biomes. For example, our first objective is to build a model that converts a rainy Minecraft landscape image into a clear Minecraft image without rain. Our dataset will be a self-made collection of images depicting a scene with rainy conditions, snowy conditions, cloudy conditions, nighttime, daytime, and with different biomes. Each image will be grouped together with other images of the same scene under different conditions. For example, a scene during the day with clear weather will be paired up with the same scene at night under rainy weather. If we have extra time, we will explore the unsupervised learning task of generating realistic Minecraft biome landscapes given unpaired images of different biomes. For example, a scene taken in the desert biome will be converted to the same image in the underwater biome, translating cacti into seaweed and sand into ocean ruins. This project will allow users to share gameplay in different environment conditions and the effectiveness of machine learning algorithms in converting between different scenery in a simulated realistic environment of Minecraft.

### Algorithms

We will build generative adversarial networks (GANs) models that utilize various convolutional and dense layers.

### Evaluation Plan

For our weather conversions, our goal for the status report, the base case for our project will be rain against a brick wall. Hence, the algorithm will be tasked with just removing the rain and not adjusting the sky and lighting. Afterwards, we will tackle simple shaped landscapes with no objects. Eventually, we will convert images involving buildings and scenery. 

Once weather conversion is acceptable, we will attempt to convert images between night and day. We will use the same base case of a brick wall to test the lighting. Then we will tackle images with the sky, since the sun, moon, stars, and clouds will have to be converted. Our last step is to convert images involving buildings and scenery. Once this is accomplished, we will use a similar procedure for biomes in terms of the complexity of scenes. 

Our first image will have the size of 256x256, relatively small images. This might pixelate the elements and force the algorithm to focus on colors. Eventually we want to use larger images that are 512x512 to capture the details. 1024x1024 would be ideal, but it could take a long time depending on the hardware. 

#### Quantitative 

Once our model predicts a clear weather image given a weather condition image input, its pixels will be compared to the real clear weather image. Our goal is for at least 70% of the pixels in the output image to be accurately matching their corresponding pixels in the clear image. The same method of pixel comparison can be applied for day/night and biome conversion. We can verify the internals of the algorithm work by pixel accuracy improving over time. We will build multiple baseline models using simple GANs and increase complexity of the models according to existing image reconstruction GANs algorithms that have been published. 

#### Qualitative 

Human participants can compare a mixed set of model output images and target clear weather images and ask if the image has been altered or processed in some way. If the image was de-rained, the participants can describe how accurately they think the produced image fares compared to clear paired image. They could describe which attributes make it look fabricated, giving an insight to what is regarded as the Minecraft aesthetic. The same would be applied for the day/night and biome conversions, further building the concept of what something out of Minecraft is supposed to look like. It would be impressive to produce results with video or live footage as well.

### Appointment with Instructor

4:30 PM on Thursday, October 22


### Group Meeting Time

7:30PM on Monday, November 2

