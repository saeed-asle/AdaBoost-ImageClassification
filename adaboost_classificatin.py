import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
#Data link : https://www.kaggle.com/puneet6060/intel-image-classification

# Define file paths
trainpath = '/Users/Saeed/Desktop/deap learing and mchine learning/all_about_machine_and_deep_learning/seg_train/'
testpath = '/Users/Saeed/Desktop/deap learing and mchine learning/all_about_machine_and_deep_learning/seg_test/'

# Class labels
code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

# Function to get the class name from the code
def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

# Image size
s = 100

# Load data and labels
X = []
y = []

for folder in os.listdir(trainpath + 'seg_train1'):
    files = glob.glob(os.path.join(trainpath + 'seg_train1', folder, '*.jpg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        image_array = image_array / 255.0  # Normalize pixel values
        X.append(image_array)
        y.append(code[folder])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,random_state=42)
y_train_one_hot = to_categorical(y_train,num_classes=6)
y_val_one_hot = to_categorical(y_val,num_classes=6)
class AdaBoostRandomForest:
    def __init__(self, n_estimators=10, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.alphas = []
    def fit(self, X, y):
        num_samples,num_classes= y.shape
        sample_weights = np.ones(num_samples)/num_samples
        for _ in range(self.n_estimators):
            estimator= RandomForestClassifier(n_estimators=50,max_depth=10)
            estimator.fit(X,y,sample_weight=sample_weights)
            predictions=estimator.predict(X)

            # Calculate weighted error
            misclassified=np.sum(np.argmax(y,axis=1)!=np.argmax(predictions, axis=1))
            weighted_error=misclassified/num_samples

            # Calculate alpha
            alpha=0.5*np.log((1-weighted_error) /(weighted_error+1e-5))
            self.estimators.append(estimator)
            self.alphas.append(alpha)

            # Update sample weights
            sample_weights*=np.exp(-alpha*misclassified)
            sample_weights/=np.sum(sample_weights)
    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.estimators[0].classes_)
        weighted_predictions = np.zeros((num_samples, num_classes))
        for i, estimator in enumerate(self.estimators):
            estimator_predictions = estimator.predict(X)
            weighted_predictions += self.alphas[i] * estimator_predictions
        predicted_labels = np.argmax(weighted_predictions, axis=1)
        return predicted_labels



ada_boost = AdaBoostRandomForest(n_estimators=100, learning_rate=0.1)

# Fit the model using one-hot encoded labels
ada_boost.fit(X_train.reshape(len(X_train), -1), y_train_one_hot)

# Predict on the validation set
predictions = ada_boost.predict(X_val.reshape(len(X_val), -1))
print(predictions)

# Ensure predictions have the correct shape
predictions = np.array(predictions)
print("Shape of y_val:", y_val_one_hot.shape)
print("Shape of predictions:", predictions.shape)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(np.argmax(y_val_one_hot, axis=1), predictions)
print("Accuracy:", accuracy)
classification_rep = classification_report(np.argmax(y_val_one_hot, axis=1), predictions, zero_division=0)
print("Classification Report:")
print(classification_rep)
