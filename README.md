# German Road Signs CNN
Convolutional Neural Network to Classify a set of German Road Signs from Small Images.

Using 2 blocks composed of 2 Convolution layers and a MaxPool layer, then 2 Fully Connected layers resolving with a softmax to 43 classes of road signs.

![CNN Diagram](https://user-images.githubusercontent.com/24580466/172717891-9e37b966-886a-4302-925f-e408e8e85f73.png)
(src:https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a)

Metrics show a converging 96% accuracy throughout training and validation sets using an adam optimizer and a learning rate of 0.0007

![Metrics_15_epochs_0 00007_LR](https://user-images.githubusercontent.com/24580466/172425167-273be5ec-5eeb-42a6-8c0d-4ccc0511fc89.png)
