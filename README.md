# Potato Disease Classification

This is a potato disease classification web app made using **TensorFlow** and **Keras' Sequential API**. It needs jpg image of a potato leaf and using Convolutional Neural Networks it predicts if the plant has:

- Early Blight
- Late Blight
- No Disease (Healthy)


##  🔨 Model Architecture
The model is as shown below: 

![ModelArchitecture](https://user-images.githubusercontent.com/72070007/137511188-02578d2e-1658-4438-9768-45d976f0c066.png)


## ⏳ Training
The model is trained on the dataset taken from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).
In addition to the following dataset, data-augmentation was also performed to significantly increase the diversity of data available.

## 🚀 Deployment
The web app is deployed using **StreamLit** and **Heroku**. Here is the link to the app:
[Potato Blight Detector](https://potato-blight-detector.herokuapp.com/)

*Note that currently only jpg image is supported to give accurate predictions.*
