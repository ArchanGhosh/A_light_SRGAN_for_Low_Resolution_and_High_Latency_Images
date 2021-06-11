# A Light SRGAN for Low Rsolution and High Latency Images

### Repository for the [A Light SRGAN For Low Resolution and High Latency Images]() Presented at [ICACDS'21](https://icacds.com/)

The model, derived from original SRGAN is supposed to work on much smaller images and with a much higher noise value. The target of the paper was to create a Model that is fast, light and trained on a much smaller dataset and time.

The paper was reviewed at 2 places before being published at ICACDS'21 along with being published in [Springer CCIS](https://www.springer.com/series/7899) Journal. 

<br>

## Abstract

In the past few years Single Image Super-Resolution(SISR)
has been one of the most researched topics in the field of AI. SuperResolution Generative Adversarial Nets in short SRGAN paved the way
to achieve Super-Resolution(SR) of images while hallucinating a lot of
details. Deriving from the main components from SRGAN, i.e. Architecture, Loss and Adversarial nature, we have refined a model that works for
very small images, and tries to make out as much information as possible
in a short amount of time. The main things being focused are to create
a fast Generator which also tries to keep a good SSIM score with the
ground truth images, tries to recover as much of the information from
relative pixels and also gets close enough to benchmark performance with
as limited resources as possible. The core objective of having a simple,
fast and light model, is not only to enlarge images but fill in as many
missing details as it can from simple pixels, to fully defined and distinct features within that image that might have double or quadruple
resolution than the Low-Resolution Images.


## Network Architecture

![Generator](https://github.com/ArchanGhosh/A_light_SRGAN_for_Low_Resolution_and_High_Latency_Images/blob/master/Network%20Architecture/Architecture-1.png)
<br>

The Full Size Generator can Be found [Here](https://github.com/ArchanGhosh/A_light_SRGAN_for_Low_Resolution_and_High_Latency_Images/blob/master/Network%20Architecture/gen_info_full.png)


![Discriminator](https://github.com/ArchanGhosh/A_light_SRGAN_for_Low_Resolution_and_High_Latency_Images/blob/master/Network%20Architecture/Discriminator-1.png)

<br>

The Full Size Discriminator can Be found [Here](https://github.com/ArchanGhosh/A_light_SRGAN_for_Low_Resolution_and_High_Latency_Images/blob/master/Network%20Architecture/dis_info_full.png)

<br>

## Dataset Used

We have used the following dataset for training:

[COIL-100](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)


[Div2k-Training Split](https://data.vision.ee.ethz.ch/cvl/DIV2K/)


And the following dataset as test:


[Set-5](https://deepai.org/dataset/set5-super-resolution)


[Set-14](https://deepai.org/dataset/set14-super-resolution)



The code are not provided in the exact manner because of copyright, but they can be easily recreated in a TF.2x Env.


Feel free to contact if you have any queries.
