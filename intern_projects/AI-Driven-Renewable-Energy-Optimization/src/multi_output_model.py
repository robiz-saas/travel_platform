# src/multi_output_model.py
"""
Multi-output neural network for simultaneous defect classification and efficiency regression
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class SolarPanelMultiOutputModel(nn.Module):
    """
    Multi-output model that predicts both defect type and efficiency percentage
    """

    def __init__(self, num_classes=6, pretrained=True):
        super(SolarPanelMultiOutputModel, self).__init__()

        # Load pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Remove the original classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Classification head (defect type)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Regression head (efficiency percentage)
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1 (will be scaled to 0-100%)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)

        # Process through shared layers
        shared_features = self.shared_layers(features)

        # Classification output (defect type)
        classification_output = self.classifier(shared_features)

        # Regression output (efficiency 0-1, will be scaled to 0-100%)
        regression_output = self.regressor(shared_features)

        return classification_output, regression_output


class SolarPanelDataset(Dataset):
    """
    Custom dataset for loading images with both classification and efficiency labels
    """

    def __init__(self, data_path, split_csv, transform=None):
        """
        Args:
            data_path: Path to enhanced solar dataset
            split_csv: CSV file with image paths and efficiency values
            transform: Image transformations
        """
        self.data_path = data_path
        self.split_data = pd.read_csv(os.path.join(data_path, split_csv))
        self.transform = transform

        # Class mapping
        self.class_names = ['bird_droppings', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        print(f"📊 Loaded {len(self.split_data)} samples from {split_csv}")

        # Print efficiency statistics
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics"""
        print(
            f"   Efficiency range: {self.split_data['efficiency'].min():.1f}% - {self.split_data['efficiency'].max():.1f}%")
        print(f"   Average efficiency: {self.split_data['efficiency'].mean():.1f}%")

        # Print class distribution
        class_counts = {}
        for _, row in self.split_data.iterrows():
            class_name = row['image_path'].split('/')[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("   Class distribution:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count} images")

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        row = self.split_data.iloc[idx]

        # Load image
        image_path = os.path.join(self.data_path, row['image_path'])
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get class label
        class_name = row['image_path'].split('/')[0]
        class_label = self.class_to_idx[class_name]

        # Get efficiency label (normalize to 0-1)
        efficiency = row['efficiency'] / 100.0  # Convert percentage to 0-1 range

        return image, class_label, efficiency


def get_transforms():
    """Get image transformations for training and validation"""

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_data_loaders(data_path, batch_size=32):
    """Create data loaders for training, validation, and testing"""

    train_transforms, val_transforms = get_transforms()

    # Create datasets
    train_dataset = SolarPanelDataset(data_path, 'train_split.csv', train_transforms)
    val_dataset = SolarPanelDataset(data_path, 'val_split.csv', val_transforms)
    test_dataset = SolarPanelDataset(data_path, 'test_split.csv', val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.class_names


class MultiOutputLoss(nn.Module):
    """
    Combined loss function for classification and regression
    """

    def __init__(self, classification_weight=1.0, regression_weight=1.0):
        super(MultiOutputLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

    def forward(self, class_pred, efficiency_pred, class_target, efficiency_target):
        # Classification loss
        cls_loss = self.classification_loss(class_pred, class_target)

        # Regression loss (efficiency)
        reg_loss = self.regression_loss(efficiency_pred.squeeze(), efficiency_target.float())

        # Combined loss
        total_loss = (self.classification_weight * cls_loss +
                      self.regression_weight * reg_loss)

        return total_loss, cls_loss, reg_loss


# Test the model architecture
if __name__ == "__main__":
    print("🧠 Testing Multi-Output Model Architecture")
    print("=" * 50)

    # Create model
    model = SolarPanelMultiOutputModel(num_classes=6)

    # Test with dummy input
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images

    with torch.no_grad():
        class_output, efficiency_output = model(dummy_input)

    print(f"✅ Model architecture test successful!")
    print(f"   📊 Input shape: {dummy_input.shape}")
    print(f"   🔍 Classification output shape: {class_output.shape}")
    print(f"   ⚡ Efficiency output shape: {efficiency_output.shape}")
    print(f"   📈 Efficiency predictions (0-1 range): {efficiency_output.squeeze().numpy()}")

    # Test loss function
    loss_fn = MultiOutputLoss()
    dummy_class_targets = torch.randint(0, 6, (4,))
    dummy_efficiency_targets = torch.rand(4)

    total_loss, cls_loss, reg_loss = loss_fn(class_output, efficiency_output,
                                             dummy_class_targets, dummy_efficiency_targets)

    print(f"   💰 Loss function test:")
    print(f"      Total loss: {total_loss.item():.4f}")
    print(f"      Classification loss: {cls_loss.item():.4f}")
    print(f"      Regression loss: {reg_loss.item():.4f}")

    print(f"\n🎯 Model ready for training!")