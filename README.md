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

**5. Key Insights from EDA**

Wheat images show clear class separability after normalization.

Visual inspection confirms high inter-class visual consistency.

Preprocessing significantly improves clarity and reduces noise in the input data.

**6. Scope of the Solution**

This model can be expanded into:

* Mobile-based applications for field use by farmers.
* Integration with drones for automated crop monitoring.
* Cloud-based dashboards for large-scale agricultural analytics.

**7. Constraints**

* Performance highly dependent on dataset quality and label accuracy.
* Limited dataset size may restrict generalization across geographies and lighting conditions.
* Computational resources: Training CNNs on large image datasets requires GPU acceleration.

**8. Stakeholders**

* Farmers: Use model insights for disease management and crop planning.
* Agricultural researchers: Enhance disease detection datasets.
* Government and NGOs: Support food security initiatives with predictive analytics tools.

**9. Tools & Libraries**

* TensorFlow / Keras
* NumPy, Pandas, Matplotlib, Seaborn
* OpenCV & scikit-image for image preprocessing
* Google Colab for development and GPU acceleration
