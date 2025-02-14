import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

# Class Labels (Ensure the order matches your training data)
class_labels = ['brain', 'hands', 'knee', 'lungs']

# Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_labels))
model.load_state_dict(torch.load('organ_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classify the Uploaded Image
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_index = predicted.item()
        confidence_percentage = confidence.item() * 100

    return class_labels[class_index], confidence_percentage

# Main Window Class
class OrganClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Organ Classifier")
        self.setGeometry(200, 200, 500, 600)
        self.setStyleSheet("background-color: #F5F5F5;")
        
        # Title Label
        title = QLabel("Organ Classifier", self)
        title.setFont(QFont('Helvetica', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333;")
        
        # Image Label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Result Labels
        self.result_label = QLabel("Predicted Class: ", self)
        self.result_label.setFont(QFont('Helvetica', 16))
        self.result_label.setStyleSheet("color: #333;")
        
        self.confidence_label = QLabel("Confidence: ", self)
        self.confidence_label.setFont(QFont('Helvetica', 14))
        self.confidence_label.setStyleSheet("color: #333;")
        
        # Upload Button
        upload_button = QPushButton("Upload Image", self)
        upload_button.setFont(QFont('Helvetica', 14))
        upload_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 5px;
        """)
        upload_button.clicked.connect(self.upload_image)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(upload_button)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def upload_image(self):
        # Select Image File
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png)")
        
        if not file_path:
            return
        
        # Display Image
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        
        # Classify Image
        predicted_class, confidence = classify_image(file_path)
        self.result_label.setText(f'Predicted Class: {predicted_class}')
        self.confidence_label.setText(f'Confidence: {confidence:.2f}%')

# Main Function
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OrganClassifierApp()
    window.show()
    sys.exit(app.exec_())
