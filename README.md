# Face Recognition using Siamese Networks

This project is a facial recognition model using Siamese Neural Networks that can identify if two images contain the same person or not. The model is trained on the [**Labeled Faces in the Wild (LFW) dataset**](http://vis-www.cs.umass.edu/lfw/) and uses data augmentation techniques to increase the accuracy of the model. The code is written in Python and uses TensorFlow and scikit-learn libraries.

---

## :pushpin: My goals :pushpin:
* Implementing & Experimenting Siamese Networks (with and without triplet loss)
* Comparing Siamese Networks with and without fine-tuning for image similarity tasks
* Creating (if good results) my own re-identification application for, as an example, an automatic facial scan to have access to restricted areas.

---

## The Siamese Network

Siamese networks Siamese Networks are commonly used for tasks related to similarity learning such as **Signature verification**, **Fraud detection** and, for our project, **Face Recognition** !

I **highly** recommend the [Andrew Ng lessons](https://youtu.be/6jfw8MuKwpI) about the Siamese Network to understand the behaviour of the SN architecture.

Here is a schema from a super [medium link](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) to understand and apply a classic Siamese Network with Tensorflow.
![Siamese Network](https://miro.medium.com/max/1400/1*dFY5gx-Vze3micJ0AMVp0A.jpeg)

There is another version of the Siamese Network but applying another input logic with anchors. Here is a link to understand the [Triplet loss algorithm](https://medium.com/analytics-vidhya/triplet-loss-b9da35be21b8)
![Siamese Network with triplet loss](https://i.ytimg.com/vi/d2XB5-tuCWU/maxresdefault.jpg)

---

## Architecture of the project

I divided the project into 2 parts:
* Implementing classic (pair) Siamese Network, with pair loss
* Implementing triplet Siamese Network with triplet loss

I first started without anchors. I was not happy with the final accuracy that I got.
So I decided to extend the basic Semantic Network algorithm to include anchors to reduce overfitting !

---