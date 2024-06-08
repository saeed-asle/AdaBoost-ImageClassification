# AdaBoost with Random Forest Base Classifier for Image Classification
Authored by saeed asle
# Description
This project demonstrates the use of AdaBoost with a Random Forest base classifier for image classification.
The dataset used is the Intel Image Classification dataset, which consists of images of various natural scenes like buildings, forests, mountains, etc.
# steps:
* Data Loading and Preprocessing: Loads images from the dataset and preprocesses them by resizing and normalizing pixel values.
* Model Definition: Defines a custom AdaBoost class that uses a Random Forest base classifier.
* Model Training: Trains the AdaBoost model on the training data.
* Model Evaluation: Evaluates the trained model on the validation data and prints the classification report.
# Features
* Data Loading and Preprocessing: Loads images from the dataset and preprocesses them by resizing and normalizing pixel values.
* Custom AdaBoost Implementation: Implements a custom AdaBoost class that uses a Random Forest base classifier for classification.
* Model Training: Trains the AdaBoost model on the training data.
* Model Evaluation: Evaluates the trained model on the validation data and prints the classification report.
# Dependencies
* os, glob: For file path operations and directory listing.
* cv2: For image reading and resizing.
* matplotlib.pyplot, numpy: For general operations and plotting.
* sklearn.metrics.accuracy_score, sklearn.metrics.classification_report: For calculating accuracy and classification report.
* sklearn.ensemble.RandomForestClassifier: For the Random Forest base classifier.
* sklearn.model_selection.train_test_split: For splitting the data into training and validation sets.
* keras.utils.to_categorical: For one-hot encoding the labels.
# How to Use
* Ensure you have the necessary libraries installed, such as opencv-python, scikit-learn, numpy, matplotlib, and keras.
* Load images from the Intel Image Classification dataset using the provided data link.
* Preprocess the images by resizing them to a common size and normalizing the pixel values.
* Define and train the AdaBoost model with a Random Forest base classifier using the training data.
* Evaluate the trained model on the validation data and print the classification report.
# Output
* The code outputs the following results:
    * Predicted labels for the validation set.
    * Accuracy score on the validation set.
    * Classification report containing precision, recall, f1-score, and support for each class.
