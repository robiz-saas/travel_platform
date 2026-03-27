# src/ai_predict.py
"""
AI-based prediction system that directly predicts both defect type and efficiency from images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from multi_output_model import SolarPanelMultiOutputModel


class AISolarPanelPredictor:
    """
    AI-powered predictor for simultaneous defect classification and efficiency regression
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the AI predictor

        Args:
            model_path (str): Path to trained multi-output model
            device: PyTorch device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = None

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load the trained model
        self.load_model()

    def load_model(self):
        """Load the trained multi-output model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Get class names and model info
            self.class_names = checkpoint['class_names']
            num_classes = len(self.class_names)

            # Initialize model architecture
            self.model = SolarPanelMultiOutputModel(num_classes=num_classes)

            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ AI model loaded successfully!")
            print(f"   📁 Model: {self.model_path}")
            print(f"   📱 Device: {self.device}")
            print(f"   🏷️ Classes: {self.class_names}")
            print(f"   📊 Training epoch: {checkpoint.get('epoch', 'Unknown')}")

            return True

        except Exception as e:
            print(f"❌ Error loading AI model: {str(e)}")
            print(f"   Please ensure you have trained the multi-output model first")
            return False

    def preprocess_image(self, image):
        """
        Preprocess image for model input

        Args:
            image: PIL Image or path to image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")

        # Apply preprocessing transforms
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict(self, image, return_probabilities=False):
        """
        AI-powered prediction of both defect type and efficiency

        Args:
            image: PIL Image or path to image
            return_probabilities (bool): Whether to return class probabilities

        Returns:
            dict: Complete prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded. Please check model path.',
                'success': False
            }

        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)

            # Make prediction
            with torch.no_grad():
                class_logits, efficiency_output = self.model(image_tensor)

                # Process classification output
                class_probabilities = F.softmax(class_logits, dim=1)
                confidence, predicted_class_idx = torch.max(class_probabilities, 1)

                predicted_class = self.class_names[predicted_class_idx.item()]
                confidence_score = confidence.item()

                # Process efficiency output (convert from 0-1 to percentage)
                predicted_efficiency = efficiency_output.item() * 100.0

                # Get all class probabilities if requested
                all_probabilities = {}
                if return_probabilities:
                    for i, class_name in enumerate(self.class_names):
                        all_probabilities[class_name] = class_probabilities[0][i].item()

                # Determine priority based on AI-predicted efficiency
                priority = self._determine_priority(predicted_efficiency, predicted_class)

                # Get AI-informed recommendations
                recommendations = self._get_ai_recommendations(predicted_class, predicted_efficiency)

                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'predicted_efficiency': round(predicted_efficiency, 1),
                    'priority': priority,
                    'recommendations': recommendations,
                    'model_type': 'AI Neural Network',
                    'success': True
                }

                if return_probabilities:
                    result['all_probabilities'] = all_probabilities

                return result

        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'success': False
            }

    def _determine_priority(self, efficiency, defect_type):
        """
        Determine maintenance priority based on AI-predicted efficiency and defect type
        """
        # High priority conditions
        if efficiency < 40:
            return 'High'
        elif defect_type in ['electrical_damage', 'physical_damage'] and efficiency < 60:
            return 'High'
        elif defect_type == 'dusty' and efficiency < 50:
            return 'High'

        # Medium priority conditions
        elif efficiency < 70:
            return 'Medium'
        elif defect_type in ['bird_droppings', 'snow_covered'] and efficiency < 80:
            return 'Medium'

        # Low priority (good condition)
        else:
            return 'Low'

    def _get_ai_recommendations(self, defect_type, efficiency):
        """
        Get AI-informed recommendations based on predicted defect and efficiency
        """
        base_recommendations = {
            'clean': [
                'Continue regular cleaning schedule',
                'Monitor for dust accumulation',
                'Optimal performance maintained'
            ],
            'dusty': [
                'Immediate cleaning required',
                'Increase cleaning frequency',
                'Consider automated cleaning systems',
                'Check environmental dust sources'
            ],
            'bird_droppings': [
                'Clean affected areas promptly',
                'Install bird deterrent systems',
                'Regular inspection for new droppings',
                'Monitor efficiency impact'
            ],
            'snow_covered': [
                'Safe snow removal when appropriate',
                'Allow natural melting when possible',
                'Optimize panel tilt for snow shedding',
                'Monitor weather conditions'
            ],
            'physical_damage': [
                'Professional inspection required',
                'Assess structural integrity',
                'Document damage for warranty',
                'Plan component replacement if needed'
            ],
            'electrical_damage': [
                'Immediate electrical inspection',
                'Contact certified technician',
                'Check inverter and connections',
                'Monitor for safety hazards'
            ]
        }

        recommendations = base_recommendations.get(defect_type, ['Contact maintenance professional'])

        # Add efficiency-based recommendations
        if efficiency < 30:
            recommendations.append(f"CRITICAL: {efficiency:.1f}% efficiency - Immediate action required")
        elif efficiency < 50:
            recommendations.append(f"LOW EFFICIENCY: {efficiency:.1f}% - Priority maintenance needed")
        elif efficiency < 70:
            recommendations.append(f"MODERATE EFFICIENCY: {efficiency:.1f}% - Schedule maintenance")
        else:
            recommendations.append(f"GOOD EFFICIENCY: {efficiency:.1f}% - Continue monitoring")

        return recommendations

    def predict_batch(self, image_paths):
        """
        Process multiple images efficiently

        Args:
            image_paths (list): List of image paths

        Returns:
            list: List of prediction results
        """
        results = []

        print(f"🔄 Processing {len(image_paths)} images with AI model...")

        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(image_paths)}")

            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)

        print(f"✅ Batch processing completed!")
        return results

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            # Load checkpoint for additional info
            checkpoint = torch.load(self.model_path, map_location='cpu')

            return {
                'model_type': 'Multi-Output Neural Network',
                'architecture': 'ResNet18 + Dual Heads',
                'capabilities': ['Defect Classification', 'Efficiency Regression'],
                'classes': self.class_names,
                'training_epoch': checkpoint.get('epoch', 'Unknown'),
                'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
                'device': str(self.device),
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }

        except Exception as e:
            return {"error": f"Could not retrieve model info: {str(e)}"}


# Example usage and testing
def test_ai_predictor():
    """Test the AI predictor"""

    print("🧪 Testing AI Solar Panel Predictor")
    print("=" * 40)

    # Model path
    model_path = "saved_models/best_multi_output_model.pth"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the multi-output model first using train_multi_output.py")
        return

    # Initialize predictor
    predictor = AISolarPanelPredictor(model_path)

    if predictor.model is None:
        print("❌ Failed to load model")
        return

    # Get model info
    model_info = predictor.get_model_info()
    print(f"\n📋 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # Test with dummy image (if no test images available)
    print(f"\n🖼️ Testing with sample prediction...")

    # Create a dummy test image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))

    # Make prediction
    result = predictor.predict(test_image, return_probabilities=True)

    if result['success']:
        print(f"✅ Test prediction successful!")
        print(f"   Defect Type: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Predicted Efficiency: {result['predicted_efficiency']:.1f}%")
        print(f"   Priority: {result['priority']}")
        print(f"   Model Type: {result['model_type']}")
    else:
        print(f"❌ Test prediction failed: {result['error']}")


if __name__ == "__main__":
    test_ai_predictor()