Image Classification Test Script

This file explains how to run the test.py Python script, designed to classify images of cherries, strawberries, and tomatoes using a pre-trained PyTorch model.

Requirements:

- Python 3.7 or higher
- PyTorch
- The final_model.pth file (pre-trained classification model)
- A folder named testdata with subfolders named cherry, strawberry, and tomato. Each subfolder should contain test images of the respective fruit.

Getting Started

Step 1: Download the Model

Download the pre-trained model file `final_model.pth` from the following link:

https://myvuwac-my.sharepoint.com/:u:/g/personal/vietomarc_myvuw_ac_nz/EXNJEKZselBEpNpepS4CptEBw8537AR6-Of7ThaNL8ezdA?e=bRb3WV

Place this file in the same directory as `test.py`.

Step 2: Prepare the Test Data

Ensure your test images are organised in the following folder structure:

testdata/
    cherry/
        image1.jpg
        image2.jpg
        ...
    strawberry/
        image1.jpg
        image2.jpg
        ...
    tomato/
        image1.jpg
        image2.jpg
        ...

Step 3: Run the Script

python test.py

The script will classify each image in the testdata folder as either cherry, strawberry, or tomato and produce a csv file containing the information of the missclassified instances.