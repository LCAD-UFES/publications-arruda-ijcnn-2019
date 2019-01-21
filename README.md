## Cross-Domain Car Detection Using Unsupervised Image-to-Image Translation: From Day to Night

[Vinicius F. Arruda](viniciusarruda.github.io), Thiago M. Paixão, [Rodrigo F. Berriel](http://rodrigoberriel.com), [Alberto F. De Souza](https://inf.ufes.br/~alberto), [Claudine Badue](https://www.inf.ufes.br/~claudine/), [Nicu Sebe](http://disi.unitn.it/~sebe/) and [Thiago Oliveira-Santos](https://www.inf.ufes.br/~todsantos/home)

<!---Published in *todo*: [DOI](https://www.google.com/)-->

#### Abstract

Coming soon !
<!---
Object detection has become a high-profile research area due to its important role in various tasks involving images. However, state-of-the-art detectors are data-driven, delegating the accuracy to the training dataset which must resemble the images in the target task. The acquisition of a dataset involves annotating images, an arduous and expensive process, generally requiring time and manual effort. Thus, a challenging scenario arises when the target domain of application has no annotated dataset available, making tasks in such situation to lean on a training dataset of a different domain.
Sharing this issue, object detection is a vital task for autonomous vehicles where the large amount of driving scenarios yields several domains of application requiring annotated data for the training process.
In this work, a method for training a car detection system with annotated data from a source domain (day images) without requiring the image annotations of the target domain (night images) is presented. 
For that, a model based on Generative Adversarial Networks (GANs) is explored to enable the generation of an artificial dataset with its respective annotations. The artificial dataset (fake dataset) is created translating images from day-time domain to night-time domain. The fake dataset is used to train a car detector model and results show that it outperforms baseline models trained on the source domain. Finally, several experiments were conducted for comparison showing that the method achieved significant and consistent improvements, increasing the detection performance in more than 10\% when compared to a baseline.
-->

---

### Source-code

#### CycleGAN

The source code used for the CycleGAN model was made publicy available by [Van Huy](https://github.com/vanhuyz/CycleGAN-TensorFlow).

#### Faster R-CNN

The source code used for the Faster R-CNN model was made publicy available by [Xinlei Chen](https://github.com/endernewton/tf-faster-rcnn).

For training the Faster R-CNN, a pre-trained resnet-101 model was used to initializate the process an can be downloaded [here](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz).

---

### Trained Models

#### CycleGAN

The trained model used in this paper is available [here](https://drive.google.com/drive/folders/17CJ5-cOK2CteZTPtRaT7rfW8oSt38CCe?usp=sharing).

#### Faster R-CNN

The trained models used in this paper are available [here](https://drive.google.com/drive/folders/1XRtExg-QGVA-DFJ1EKf8L0GLCxe5wIqH?usp=sharing).

---

### Dataset

#### Berkeley Deep Drive Dataset

##### Dataset Acquisition

Download the Berkeley Deep Drive dataset [here](https://bdd-data.berkeley.edu/).
Is only necessary to download the Images and Labels files.

##### Dataset Filtering

After downloading the BDD dataset, the Images and Labels will be placed into the zipped files `bdd100k_images.zip` and `bdd100k_labels.zip` respectively. In the same directory, place the provided source code `filter_dataset.py` from this repository with the folder `lists`.

On the terminal, run: `python filter_dataset.py`.
It will take a few minutes, and at the end, the folder `images` and `labels` will contain the images and bounding boxes of the images respectively. 

#### Generated (Fake) Dataset

Available [here](https://drive.google.com/drive/folders/1ZoXfgpTT1N5eOsI4-Tcv0id3mqij5gsP?usp=sharing).

---

### Videos

Videos demonstrating the inference performed by the trained Faster R-CNN model which yielded the best results with our proposed system.

#### Testing on Day+Night Dataset

Inferences performed on day+night dataset:

[![Video1](https://github.com/viniciusarruda/cross-domain-car-detection/blob/master/images/day_plus_night_video_overview.png)](https://youtu.be/qENxVuUXa0s)

#### Testing on Night Dataset

Inferences performed on night dataset:

[![Video2](https://github.com/viniciusarruda/cross-domain-car-detection/blob/master/images/night_video_overview.png)](https://youtu.be/MqZ2I-h_FOA)

---

### BibTeX

todo

<!--
    @article{berriel2017grsl,
        Author  = {Rodrigo F. Berriel and Andre T. Lopes and Alberto F. de Souza and Thiago Oliveira-Santos},
        Title   = {{Deep Learning Based Large-Scale Automatic Satellite Crosswalk Classification}},
        Journal = {IEEE Geoscience and Remote Sensing Letters},
        Year    = {2017},
        DOI     = {10.1109/LGRS.2017.2719863},
        ISSN    = {1545-598X},
    }
-->
