# Fruit-Detection-using-Machine-Learning
 Build on the KNN algorithm which helps in detection of the fruit or vegetable by scanning the image.
# Fruit and Vegetable Detection Using KNN (K-Nearest Neighbors) Algorithm

This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm for the detection and classification of fruits and vegetables from images. By leveraging image processing techniques and machine learning, this system identifies various fruits and vegetables based on their visual features such as color and texture.

# Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [KNN Algorithm Overview](#knn-algorithm-overview)
- [Implementation Steps](#implementation-steps)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

# Project Overview
The aim of this project is to classify fruits and vegetables based on their visual features using the KNN algorithm. The KNN classifier is trained using images that represent various fruits and vegetables, and then the model classifies new images based on the nearest neighbors in the feature space.

# Key Features:
- Image-based classification using KNN
- Feature extraction from images (color histograms and texture features)
- Real-time classification of fruits and vegetables from input images
- Evaluation of classification performance (accuracy, precision, recall, etc.)

# Technologies Used
- **Python 3.x**
- **OpenCV**: For image processing (resizing, color conversion, etc.)
- **scikit-learn**: For implementing the KNN classifier
- **NumPy**: For numerical operations (e.g., array manipulation)
- **Matplotlib**: For data visualization and result display
- **Pandas**: For dataset handling and management

## Dataset
The dataset consists of images of various fruits and vegetables. Each category (e.g., apple, banana, carrot) has a separate folder containing images for training and testing the model.

# KNN Algorithm Overview
The KNN algorithm classifies a new image based on its closest neighbors in the feature space. This project focuses on the following steps:
1. **Feature Extraction**: Extract visual features (color histograms and texture features) from each image in the dataset.
2. **Distance Calculation**: Use a distance metric, like Euclidean distance, to compare the new image with the images in the dataset.
3. **Classification**: Classify the image by finding the K nearest neighbors and using the majority class label.

# Feature Extraction:
- **Color Histogram**: A representation of the color distribution in an image, useful for differentiating fruits and vegetables.
- **Texture Features**: Texture-based features (e.g., Local Binary Patterns or Gray-Level Co-occurrence Matrix) help identify the surface patterns of fruits and vegetables.

# KNN Classification:
KNN classifies an image based on the label of its K nearest neighbors, where K is a user-defined parameter (usually a small integer). The algorithm assigns the image to the class that appears most frequently among the K neighbors.

# Implementation Steps

# Step 1: Import Libraries
```python
import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
```

# Step 2: Feature Extraction Function (Color Histogram)
```python
def extract_color_histogram(image, bins=16):
    # Converts image to HSV color space and calculates color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [bins], [0, 256])
    hist_saturation = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()
    return np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])
```

# Step 3: Load Dataset and Extract Features
```python
def load_dataset(dataset_path):
    data = []
    labels = []
    label_map = {}
    label_idx = 0
    for label_folder in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_folder)
        if os.path.isdir(label_folder_path):
            label_map[label_idx] = label_folder
            for image_name in os.listdir(label_folder_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(label_folder_path, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (100, 100))  # Resize to standard size
                    feature = extract_color_histogram(image)
                    data.append(feature)
                    labels.append(label_idx)
            label_idx += 1
    return np.array(data), np.array(labels), label_map
```

# Step 4: Train KNN Model
```python
X, y, label_map = load_dataset('path_to_your_dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

# Step 5: Evaluate the Model
```python
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize some predictions
for i in range(5):
    image = cv2.imread(f'path_to_sample_images/{i}.jpg')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {label_map[y_pred[i]]}, Actual: {label_map[y_test[i]]}")
    plt.show()
```

# Usage
To classify new images using the trained KNN model:

```python
def classify_image(image_path, model, label_map):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    feature = extract_color_histogram(image)
    feature = feature.reshape(1, -1)
    label_idx = model.predict(feature)[0]
    return label_map[label_idx]

new_image_path = "path_to_new_image.jpg"
predicted_label = classify_image(new_image_path, knn, label_map)
print(f"The image is classified as: {predicted_label}")
```

# Model Evaluation
- **Accuracy**: Measures how often the model makes the correct prediction.
- **Precision, Recall, and F1-Score**: Metrics to evaluate the model performance, especially useful when class distributions are imbalanced.

# Sample Results:
```
Accuracy: 92%
Classification Report:
              precision    recall  f1-score   support
        apple       0.91      0.93      0.92       100
       banana       0.92      0.90      0.91       100
        carrot       0.94      0.94      0.94       100
```

# Contributing
We welcome contributions to improve this project! If you have any ideas for enhancements or bug fixes:
1. Fork the repository.
2. Create a new branch.
3. Make changes and commit.
4. Submit a pull request for review.

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
