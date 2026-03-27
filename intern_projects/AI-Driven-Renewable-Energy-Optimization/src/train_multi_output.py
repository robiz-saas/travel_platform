# src/train_multi_output.py - FIXED VERSION
"""
Training pipeline for multi-output neural network
CORRECTED: Fixed all imports, dependencies, and code issues
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try importing optional visualization libraries
try:
    import matplotlib

    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    print("⚠️ Warning: matplotlib/seaborn not available. Training will continue without plots.")
    HAS_PLOTTING = False

try:
    from sklearn.metrics import classification_report, confusion_matrix

    HAS_SKLEARN = True
except ImportError:
    print("⚠️ Warning: scikit-learn not available. Will skip some evaluation metrics.")
    HAS_SKLEARN = False

# Import our custom modules
from multi_output_model import (
    SolarPanelMultiOutputModel,
    MultiOutputLoss,
    create_data_loaders
)


class MultiOutputTrainer:
    """
    Trainer class for multi-output solar panel model
    """

    def __init__(self, data_path, device=None, batch_size=32):
        self.data_path = data_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        print(f"🚀 Initializing trainer...")
        print(f"   📱 Device: {self.device}")
        print(f"   📊 Batch size: {batch_size}")

        # Create data loaders
        try:
            self.train_loader, self.val_loader, self.test_loader, self.class_names = create_data_loaders(
                data_path, batch_size
            )
        except Exception as e:
            print(f"❌ Error creating data loaders: {str(e)}")
            print(f"   Make sure {data_path} contains train_split.csv, val_split.csv, test_split.csv")
            raise

        # Initialize model
        self.model = SolarPanelMultiOutputModel(num_classes=len(self.class_names))
        self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = MultiOutputLoss(classification_weight=1.0, regression_weight=10.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        # Training history
        self.history = {
            'train_total_loss': [], 'train_cls_loss': [], 'train_reg_loss': [],
            'val_total_loss': [], 'val_cls_loss': [], 'val_reg_loss': [],
            'train_cls_acc': [], 'val_cls_acc': [],
            'train_efficiency_mae': [], 'val_efficiency_mae': []
        }

        print(f"✅ Trainer initialized successfully!")
        print(f"   🏷️ Classes: {len(self.class_names)}")
        print(f"   📁 Training samples: {len(self.train_loader.dataset)}")
        print(f"   📁 Validation samples: {len(self.val_loader.dataset)}")
        print(f"   📁 Test samples: {len(self.test_loader.dataset)}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        correct_predictions = 0
        total_samples = 0
        efficiency_mae_sum = 0.0

        for batch_idx, (images, class_labels, efficiency_labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            efficiency_labels = efficiency_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            class_pred, efficiency_pred = self.model(images)

            # Calculate loss
            total_loss_batch, cls_loss_batch, reg_loss_batch = self.criterion(
                class_pred, efficiency_pred, class_labels, efficiency_labels
            )

            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            cls_loss_sum += cls_loss_batch.item()
            reg_loss_sum += reg_loss_batch.item()

            # Classification accuracy
            _, predicted_classes = torch.max(class_pred, 1)
            correct_predictions += (predicted_classes == class_labels).sum().item()
            total_samples += class_labels.size(0)

            # Efficiency MAE (convert back to percentage)
            efficiency_mae = torch.abs(efficiency_pred.squeeze() - efficiency_labels).mean() * 100
            efficiency_mae_sum += efficiency_mae.item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                current_acc = 100 * correct_predictions / total_samples if total_samples > 0 else 0
                print(f"   Batch {batch_idx:3d}/{len(self.train_loader)}: "
                      f"Loss: {total_loss_batch.item():.4f}, "
                      f"Cls Acc: {current_acc:.1f}%, "
                      f"Eff MAE: {efficiency_mae.item():.1f}%")

        # Calculate epoch metrics
        avg_total_loss = total_loss / len(self.train_loader)
        avg_cls_loss = cls_loss_sum / len(self.train_loader)
        avg_reg_loss = reg_loss_sum / len(self.train_loader)
        cls_accuracy = 100 * correct_predictions / total_samples
        efficiency_mae = efficiency_mae_sum / len(self.train_loader)

        return avg_total_loss, avg_cls_loss, avg_reg_loss, cls_accuracy, efficiency_mae

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        correct_predictions = 0
        total_samples = 0
        efficiency_mae_sum = 0.0

        all_class_preds = []
        all_class_labels = []
        all_efficiency_preds = []
        all_efficiency_labels = []

        with torch.no_grad():
            for images, class_labels, efficiency_labels in self.val_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                efficiency_labels = efficiency_labels.to(self.device)

                # Forward pass
                class_pred, efficiency_pred = self.model(images)

                # Calculate loss
                total_loss_batch, cls_loss_batch, reg_loss_batch = self.criterion(
                    class_pred, efficiency_pred, class_labels, efficiency_labels
                )

                # Accumulate metrics
                total_loss += total_loss_batch.item()
                cls_loss_sum += cls_loss_batch.item()
                reg_loss_sum += reg_loss_batch.item()

                # Classification accuracy
                _, predicted_classes = torch.max(class_pred, 1)
                correct_predictions += (predicted_classes == class_labels).sum().item()
                total_samples += class_labels.size(0)

                # Efficiency MAE
                efficiency_mae = torch.abs(efficiency_pred.squeeze() - efficiency_labels).mean() * 100
                efficiency_mae_sum += efficiency_mae.item()

                # Store predictions for detailed analysis
                all_class_preds.extend(predicted_classes.cpu().numpy())
                all_class_labels.extend(class_labels.cpu().numpy())
                all_efficiency_preds.extend((efficiency_pred.squeeze() * 100).cpu().numpy())
                all_efficiency_labels.extend((efficiency_labels * 100).cpu().numpy())

        # Calculate epoch metrics
        avg_total_loss = total_loss / len(self.val_loader)
        avg_cls_loss = cls_loss_sum / len(self.val_loader)
        avg_reg_loss = reg_loss_sum / len(self.val_loader)
        cls_accuracy = 100 * correct_predictions / total_samples
        efficiency_mae = efficiency_mae_sum / len(self.val_loader)

        return (avg_total_loss, avg_cls_loss, avg_reg_loss, cls_accuracy, efficiency_mae,
                all_class_preds, all_class_labels, all_efficiency_preds, all_efficiency_labels)

    def train(self, num_epochs=50, save_dir="saved_models"):
        """Train the multi-output model"""

        print(f"\n🏋️ Starting training for {num_epochs} epochs...")
        print("=" * 60)

        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15

        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Training
            train_metrics = self.train_epoch()
            train_total_loss, train_cls_loss, train_reg_loss, train_cls_acc, train_eff_mae = train_metrics

            # Validation
            val_metrics = self.validate_epoch()
            val_total_loss, val_cls_loss, val_reg_loss, val_cls_acc, val_eff_mae = val_metrics[:5]

            # Update learning rate
            self.scheduler.step(val_total_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_total_loss'].append(train_total_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_reg_loss'].append(train_reg_loss)
            self.history['train_cls_acc'].append(train_cls_acc)
            self.history['train_efficiency_mae'].append(train_eff_mae)

            self.history['val_total_loss'].append(val_total_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['val_reg_loss'].append(val_reg_loss)
            self.history['val_cls_acc'].append(val_cls_acc)
            self.history['val_efficiency_mae'].append(val_eff_mae)

            # Print epoch results
            print(f"🏋️ Train - Total Loss: {train_total_loss:.4f}, "
                  f"Cls Acc: {train_cls_acc:.1f}%, Eff MAE: {train_eff_mae:.1f}%")
            print(f"✅ Val   - Total Loss: {val_total_loss:.4f}, "
                  f"Cls Acc: {val_cls_acc:.1f}%, Eff MAE: {val_eff_mae:.1f}%")
            print(f"📈 Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0

                # Save model
                model_path = os.path.join(save_dir, 'best_multi_output_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'class_names': self.class_names,
                    'history': self.history,
                    'train_accuracy': train_cls_acc,
                    'val_accuracy': val_cls_acc,
                    'train_efficiency_mae': train_eff_mae,
                    'val_efficiency_mae': val_eff_mae
                }, model_path)

                print(f"💾 Best model saved! Val Loss: {best_val_loss:.4f}")

            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{max_patience}")

            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"\n🎉 Training completed!")
        print(f"💾 Best model saved at: {os.path.join(save_dir, 'best_multi_output_model.pth')}")

        return self.history

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not HAS_PLOTTING:
            print("⚠️ Plotting not available. Install matplotlib and seaborn to generate plots.")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Multi-Output Model Training History', fontsize=16)

            # Total Loss
            axes[0, 0].plot(self.history['train_total_loss'], label='Train', color='blue')
            axes[0, 0].plot(self.history['val_total_loss'], label='Validation', color='red')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Classification Accuracy
            axes[0, 1].plot(self.history['train_cls_acc'], label='Train', color='blue')
            axes[0, 1].plot(self.history['val_cls_acc'], label='Validation', color='red')
            axes[0, 1].set_title('Classification Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Efficiency MAE
            axes[1, 0].plot(self.history['train_efficiency_mae'], label='Train', color='blue')
            axes[1, 0].plot(self.history['val_efficiency_mae'], label='Validation', color='red')
            axes[1, 0].set_title('Efficiency Mean Absolute Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Loss Components
            axes[1, 1].plot(self.history['train_cls_loss'], label='Train Cls Loss', alpha=0.7)
            axes[1, 1].plot(self.history['train_reg_loss'], label='Train Reg Loss', alpha=0.7)
            axes[1, 1].plot(self.history['val_cls_loss'], label='Val Cls Loss', alpha=0.7)
            axes[1, 1].plot(self.history['val_reg_loss'], label='Val Reg Loss', alpha=0.7)
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Training plots saved to: {save_path}")
            else:
                plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
                print(f"📊 Training plots saved to: training_history.png")

            plt.close()  # Close to free memory

        except Exception as e:
            print(f"⚠️ Error creating plots: {str(e)}")


# Quick fix for train_multi_output.py - Update the main() function

def main():
    """Main training function"""

    print("🚀 Multi-Output Solar Panel Model Training")
    print("=" * 50)

    # FIXED: Use absolute path instead of relative path
    DATA_PATH = r"C:\Users\Adm\Desktop\Kalpana\python projects\solar_defects_classification\data\enhanced_solar_dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    SAVE_DIR = "saved_models"

    print(f"📁 Looking for dataset at: {DATA_PATH}")

    # Check if enhanced dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Enhanced dataset not found at: {DATA_PATH}")
        print(f"Please run prepare_efficiency_dataset.py first to create the enhanced dataset")
        return

    # Check for required CSV files with absolute paths
    required_files = ['train_split.csv', 'val_split.csv', 'test_split.csv', 'efficiency_mapping.csv']
    missing_files = []

    print(f"🔍 Checking for required files:")
    for file in required_files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"   Checking: {file_path}")
        if not os.path.exists(file_path):
            missing_files.append(file)
            print(f"   ❌ Missing: {file}")
        else:
            # Check if file has content
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                print(f"   ✅ Found: {file} ({len(df)} records)")
            except Exception as e:
                print(f"   ⚠️ Found but error reading {file}: {str(e)}")
                missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files in {DATA_PATH}:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"Please run prepare_efficiency_dataset.py to generate these files")
        return

    try:
        # Initialize trainer
        trainer = MultiOutputTrainer(
            data_path=DATA_PATH,
            batch_size=BATCH_SIZE
        )

        # Train model
        print(f"\n🏋️ Starting training...")
        history = trainer.train(num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR)

        # Plot training history
        print(f"\n📊 Creating training plots...")
        plot_path = os.path.join(SAVE_DIR, 'training_history.png')
        trainer.plot_training_history(save_path=plot_path)

        # Save training summary
        summary = {
            'training_config': {
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'data_path': DATA_PATH,
                'device': str(trainer.device)
            },
            'final_results': {
                'best_val_loss': min(history['val_total_loss']),
                'final_train_accuracy': history['train_cls_acc'][-1],
                'final_val_accuracy': history['val_cls_acc'][-1],
                'final_train_mae': history['train_efficiency_mae'][-1],
                'final_val_mae': history['val_efficiency_mae'][-1]
            },
            'class_names': trainer.class_names
        }

        summary_path = os.path.join(SAVE_DIR, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎉 Training completed successfully!")
        print(f"   📊 Final Classification Accuracy: {summary['final_results']['final_val_accuracy']:.2f}%")
        print(f"   ⚡ Final Efficiency MAE: {summary['final_results']['final_val_mae']:.2f}%")
        print(f"   💾 Best model: {os.path.join(SAVE_DIR, 'best_multi_output_model.pth')}")
        print(f"   📋 Summary: {summary_path}")
        print(f"   📊 Plots: {plot_path}")

    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()