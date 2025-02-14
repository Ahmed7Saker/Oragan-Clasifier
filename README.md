# Organ Classifier Using Transfer Learning

## Project Overview
This project is a **Deep Learning-based Organ Classifier** built using PyTorch and Transfer Learning. It classifies medical images into the following categories:
- **Brain**
- **Hands**
- **Knee**
- **Lungs**

The model utilizes **ResNet18** as a pre-trained backbone for feature extraction, followed by a customized fully connected layer for classification into four categories.

## Features
- **Transfer Learning with ResNet18**: Utilizes a pre-trained ResNet18 model for faster and more accurate training.
- **Data Augmentation**: Includes multiple data augmentation techniques like:
  - Random Horizontal Flip
  - Random Rotation
  - Random Resized Crop
  - Color Jittering
  
![Organ Classifier Overview](IMGES\Screenshot 2025-02-13 114225.png)


- **GPU Support**: Automatically detects and utilizes GPU if available for faster training.
- **Checkpointing**: Saves the model with the best validation accuracy as `organ_classifier.pth`.
- **Validation Accuracy Display**: Displays the validation accuracy after each epoch.


## Project Structure
```
Organ-Classifier/
│
├── training/                     # Training dataset
├── val/                          # Validation dataset
├── testing/                      # Testing dataset (optional)
│
├── Classifier.py                  # Main Python script for training and evaluation
├── ModelTraining.ipynb            # Jupyter Notebook version of the training script
├── organ_classifier.pth           # Saved model state with the best validation accuracy
│
└── venv/                          # Virtual environment (Not included in version control)
```

## Requirements
To install all the required libraries, use the following command:
```bash
pip install torch torchvision
```
Or, if you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Data Requirements
- The dataset is organized into three directories: `training`, `val`, and `testing`.
- The images are expected to be sorted into subdirectories for each class.

## Model Architecture
- **Fully Connected Layer**: Adjusted to output 4 classes (Brain, Hands, Knee, Lungs)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam with Learning Rate of 0.001

## How to Use
1. **Clone the Repository:**
```bash
git clone < https://github.com/Ahmed7Saker/Oragan-Clasifier.git >
git branch -M main
git push -u origin main>
```
2. **Create Virtual Environment (Optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```
3. **Install Requirements:**
```bash
pip install -r requirements.txt
```
4. **Train the Model:**
```bash
python Classifier.py
```

## License
This project is public please fork it an contripute

## Contact
For any inquiries, please contact **Ahmed Saker**.
