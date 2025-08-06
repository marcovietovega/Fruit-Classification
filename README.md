# Fruit Classification with Deep Learning

This project uses advanced techniques to accurately identify cherries, strawberries, and tomatoes in photos. By leveraging deep learning, it analyzes patterns in fruit images to classify them with precision.

### Key Features of This Project

- **Three Models**: Started simple and built up to advanced models.
- **Smart Data Processing**: Improved training data by rotating and changing images.
- **Automatic Fine-Tuning**: Found the best settings automatically.
- **Reliable Testing**: Used cross-validation to ensure trustworthy results.
- **Detailed Results**: Shows how well the model performs with charts and numbers.

## Dataset

The dataset contains photos of cherries, strawberries, and tomatoes. Images are resized to 384x384 pixels and improved by rotating, flipping, and adjusting colors. The data is split into training, validation, and test sets.

## How the Model Works

The final model uses **EfficientNetV2-S**, a pre-trained model that recognizes patterns in images. Custom layers were added to classify fruits:

- Starts with a model trained on millions of images.
- Adds layers to classify cherries, strawberries, and tomatoes.
- Uses dropout to prevent overfitting.

### The Three Models

1. **Simple Neural Network (MLP)**: A basic starting point.
2. **Convolutional Neural Network (CNN)**: Better at understanding images.
3. **EfficientNetV2**: The most advanced model using transfer learning.

## Tools and Techniques

### Libraries

- **PyTorch**: Main deep learning framework.
- **torchvision**: Tools for working with images.
- **Optuna**: Finds the best settings for the model.
- **scikit-learn**: Calculates performance scores.
- **PIL/OpenCV**: Processes and loads images.

### Techniques

- **Automatic tuning**: Finds the best settings.
- **Cross-validation**: Tests the model multiple ways.
- **Data augmentation**: Creates more training examples.
- **Transfer learning**: Uses knowledge from pre-trained models.

## Results

- **High Accuracy**: The model achieves 99% accuracy.
- **Detailed Metrics**: Precision, recall, and F1-score are all 0.99 across cherry, tomato, and strawberry classes.
- **Reliable Performance**: Cross-validation ensures consistent results.
- **Thorough Testing**: Evaluated on new images the model has never seen.

## What This Project Shows

### Data Handling

- **Smart data loading**: Loads and processes fruit images.
- **Error checking**: Ensures files and folders are set up correctly.
- **Data improvements**: Modifies images to create more training examples.
- **Balanced training**: Ensures each fruit type gets equal attention.

### Model Training

- **Step-by-step improvement**: Built three models, each better than the last.
- **Automatic optimization**: Found the best settings automatically.
- **Reliable testing**: Cross-validation ensures the model works on new data.
- **Fast training**: Works on both GPU and regular computers.

### Results and Analysis

- **Clear metrics**: Shows accuracy, precision, recall, and F1-scores.
- **Error analysis**: Confusion matrices show mistakes.
- **Mistake reports**: Lists misclassified images.
- **Training charts**: Graphs show how the model improved.

## What I Learned

This project demonstrates important machine learning skills:

- **Transfer Learning**: Using trained models to solve new problems.
- **Automatic Tuning**: Finding the best settings automatically.
- **Proper Testing**: Ensuring the model works on new data.
- **Data Augmentation**: Making training data more diverse.

## Contact

Marco Vieto Vega - [vietomarc@myvuw.ac.nz](mailto:vietomarc@myvuw.ac.nz)

---

_This project was developed as part of AIML421 coursework at Victoria University of Wellington._
