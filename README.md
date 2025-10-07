**Overview**

This project develops a deep learning model to detect wheat and classify potential diseases from image data.
The goal is to provide an automated, reliable system that assists farmers in improving crop management, yield prediction, and disease control.

**1. Problem Statement**

Farmers face challenges in timely identifying wheat diseases, which directly affect productivity.
This project aims to:

 * Build a Convolutional Neural Network (CNN) capable of classifying wheat images into 5 distinct categories.
 
 * Develop a scalable image classification pipeline using TensorFlow/Keras.
 
 * Achieve a robust accuracy suitable for real-world agricultural use.

**2. Deep Learning Approach**

**Model Type**

A tf.keras.Sequential model is used, composed of convolutional and pooling layers that progressively extract spatial features from images.

**Model Architecture**

The model consists of:

* Input rescaling layer – normalizes pixel values (Rescaling(1./255)).

* Three convolution blocks: each with Conv2D followed by MaxPooling2D.

* Flattening layer – converts 2D feature maps into a 1D feature vector.

* Dense layer with 128 units and ReLU activation.

* Output layer – 5 neurons (one per class) with softmax activation.

**Compilation Details**

Optimizer: Adam

Loss Function: Categorical Crossentropy

Evaluation Metric: Accuracy

**3. Dataset & Preprocessing**

The dataset is sourced from https://www.kaggle.com/competitions/global-wheat-detection/data and stored in google drive. 

Images are loaded using tf.keras.utils.image_dataset_from_directory(), which automatically infers labels from subdirectories.

Data normalization ensure robustness against variations in lighting and orientation.

Images are resized to a uniform shape compatible with the CNN input layer.

**4. Training & Performance**

**Training Configuration**

Model trained for 3 epochs on the dataset.

Batch size and learning rate tuned empirically.

Validation split used to monitor model generalization.

The model demonstrates strong learning performance and excellent validation accuracy, indicating effective feature extraction and generalization.
