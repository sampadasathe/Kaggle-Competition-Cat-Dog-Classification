# Kaggle-Competition: Cat Dog Classification using Google Colab
Kaggle Competition: Dogs vs. Cats Redux: Kernels Edition with 99% classification accuracy
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview

Since Cat Dog Classification is an image classification problem, it can be solved using Transfer Learning. I have used pretrained VGG16 as a classifier because it achieves top-5 accuracy of 92.3% on imagenet. Also, since the VGG16 architecture is not very complex, its a good basic model to learn about convolutional networks and image classification in general.

## Transfer Learning
In computer vision, transfer learning is usually expressed through the use of pre-trained models. A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature (e.g. VGG, Inception, MobileNet). A comprehensive review of pre-trained models’ performance on computer vision problems using data from the ImageNet (Deng et al. 2009) challenge is presented by Canziani et al. (2016).

## VGG16 model
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

## VGG16 Architecture
The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.  layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.

Three Fully-Connected (FC) layers follow a stack of convolutional layers (which has a different depth in different architectures): the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

All hidden layers are equipped with the rectification (ReLU) non-linearity. It is also noted that none of the networks (except for one) contain Local Response Normalisation (LRN), such normalization does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time.
![VGG16 Architecture](https://github.com/sampadasathe/Kaggle-Competition-Cat-Dog-Classification/blob/master/vgg16-neural-network.jpg)

## Data Augmentation
The simplest way to reduce overfitting is to increase the size of the training data. In machine learning, we were not able to increase the size of training data as the labeled data was too costly.

But, now let’s consider we are dealing with images. In this case, there are a few ways of increasing the size of the training data – rotating the image, flipping, scaling, shifting, etc. This technique is known as data augmentation. This usually provides a big leap in improving the accuracy of the model. It can be considered as a mandatory trick in order to improve our predictions.

## Improve model accuracy by training last 4 layers and using regularization
Model accuracy improved when I trained the last 4 layers of VGG16. In deep learning, regularization actually penalizes the weight matrices of the nodes.
In L2, we have:
![L2 Regularization](https://github.com/sampadasathe/Kaggle-Competition-Cat-Dog-Classification/blob/master/L2 regularization.jpg)


Here, lambda is the regularization parameter. It is the hyperparameter whose value is optimized for better results. L2 regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).

#### References
1. https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
2. https://neurohive.io/en/popular-networks/vgg16/
3. https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
4. https://medium.com/udacity-pytorch-challengers/why-use-a-pre-trained-model-rather-than-creating-your-own-d0e3a17e202f
5. https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
