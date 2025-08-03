# Fruit Classification with Deep Learning

A machine learning project that can tell the difference between cherries, strawberries, and tomatoes in photos. Built using PyTorch and modern deep learning methods.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 What This Project Does

This project creates a computer program that can look at fruit photos and correctly identify whether they show cherries, strawberries, or tomatoes. It uses deep learning to learn patterns from thousands of example images.

### What Makes This Project Special

- **Three Different AI Models**: Started simple and built up to advanced models
- **Smart Data Processing**: Made the training data better by rotating and changing images
- **Automatic Fine-Tuning**: Used Optuna to automatically find the best settings
- **Reliable Testing**: Used 5-fold cross-validation to make sure results are trustworthy
- **Detailed Results**: Shows exactly how well the model performs with charts and numbers

## 📊 Dataset

**Note**: The dataset was provided as part of the AIML421 course and is not included in this repository.

- **What it contains**: Photos of cherries, strawberries, and tomatoes
- **Image size**: All photos are resized to 384x384 pixels
- **Data improvements**: Images are rotated, flipped, and color-adjusted to help the model learn better
- **How it's split**: 80% for training, 20% for validation, plus a separate test set

To run this project, you'll need to provide your own fruit image dataset organized in folders by fruit type.

## 🏗️ How the AI Model Works

The final model uses **EfficientNetV2-S**, which is a powerful pre-trained model that already knows how to recognize patterns in images. I added my own layers on top to specifically recognize fruits:

- Starts with a model already trained on millions of images
- Adds custom layers to classify the three fruit types
- Uses dropout to prevent overfitting (memorizing instead of learning)

### The Three Models I Built

1. **Simple Neural Network (MLP)**: A basic starting point
2. **Convolutional Neural Network (CNN)**: Better at understanding images
3. **EfficientNetV2**: The most advanced model using transfer learning

## 🔧 Tools and Technologies Used

### Main Libraries

- **PyTorch**: The main deep learning framework
- **torchvision**: Tools for working with images
- **Optuna**: Automatically finds the best settings for the model
- **scikit-learn**: Calculates performance scores
- **PIL/OpenCV**: Processes and loads images

### Smart Techniques Used

- **Automatic tuning**: Finds the best learning rate, batch size, and optimizer
- **Cross-validation**: Tests the model 5 different ways to make sure it's reliable
- **Data augmentation**: Creates more training examples by modifying existing images
- **Transfer learning**: Uses knowledge from a model trained on millions of images

## 📈 Results

- **High Accuracy**: The model performs very well after automatic tuning
- **Reliable Performance**: 5-fold cross-validation ensures consistent results
- **Thorough Testing**: Detailed evaluation on completely new images the model has never seen

## 🚀 How to Use This Project

### Requirements

You'll need to install these Python packages:

```bash
pip install torch torchvision
pip install optuna scikit-learn
pip install pillow opencv-python
pip install matplotlib seaborn pandas
pip install tqdm numpy
```

### To Train a New Model

1. Open `train.ipynb` in Jupyter Notebook or Google Colab
2. Update the data paths to point to your fruit image folders
3. Run all the cells from top to bottom
4. The notebook will automatically find the best settings and train the model

### To Test the Model

1. Download the trained model file (link in Readme.txt)
2. Put your test images in folders: `testdata/cherry/`, `testdata/strawberry/`, `testdata/tomato/`
3. Run: `python test.py`
4. The script will tell you how accurate the model is and create a report of any mistakes

### Files in This Project

```
├── train.ipynb          # Main notebook with all the training code
├── test.py             # Script to test the trained model
├── final_model.pth     # The trained model (download separately)
├── README.md           # This file
├── Readme.txt          # Instructions for running the test
└── testdata/           # Folder for test images (not included)
    ├── cherry/         # Put cherry images here
    ├── strawberry/     # Put strawberry images here
    └── tomato/         # Put tomato images here
```

## 📋 What This Project Shows

### Data Handling

- **Smart data loading**: Custom code to load and process fruit images
- **Error checking**: Makes sure all files and folders are set up correctly
- **Data improvements**: Creates more training examples by modifying images
- **Balanced training**: Makes sure each fruit type gets equal attention

### Model Training

- **Step-by-step improvement**: Built three different models, each better than the last
- **Automatic optimization**: Used Optuna to find the best settings without manual guessing
- **Reliable testing**: Cross-validation ensures the model works on new data
- **Fast training**: Works on both GPU and regular computers

### Results and Analysis

- **Clear metrics**: Shows accuracy, precision, recall, and F1-scores
- **Error analysis**: Confusion matrices show exactly what mistakes the model makes
- **Mistake reports**: Lists which specific images were misclassified
- **Training charts**: Visual graphs showing how the model improved over time

## 🎓 What I Learned

This project demonstrates important machine learning skills:

- **Transfer Learning**: How to use existing trained models to solve new problems
- **Automatic Tuning**: How to let the computer find the best settings
- **Proper Testing**: How to make sure a model will work on new, unseen data
- **Data Augmentation**: How to make training data more diverse and useful
- **Professional Code**: How to write clean, well-documented code that others can use

## 📧 Contact

Marco Vieto - [vietomarc@myvuw.ac.nz](mailto:vietomarc@myvuw.ac.nz)

---

_This project was developed as part of AIML421 coursework at Victoria University of Wellington._
