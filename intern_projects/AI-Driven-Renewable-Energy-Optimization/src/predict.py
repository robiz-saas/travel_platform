# src/efficiency_predictor.py - Updated with better loss handling
"""
Solar Panel Efficiency Predictor
Based on analysis of 603 real efficiency measurements
"""

import numpy as np
from typing import Dict, Optional


class EfficiencyPredictor:
    """Predicts solar panel efficiency based on defect classification results."""

    def __init__(self):
        # Based on your actual efficiency data analysis
        self.efficiency_models = {
            'clean': {
                'expected_efficiency': 71.2,
                'std_deviation': 23.3,
                'range': (5, 95),
                'most_common': '90-100%',
                'description': 'Clean panels with optimal performance',
                'priority': 'Low'
            },
            'dusty': {
                'expected_efficiency': 35.7,
                'std_deviation': 27.1,
                'range': (5, 95),
                'most_common': '0-10%',
                'description': 'Dust significantly reduces efficiency',
                'priority': 'High'
            },
            'bird_droppings': {
                'expected_efficiency': 61.5,
                'std_deviation': 29.7,
                'range': (5, 95),
                'most_common': '80-90%',
                'description': 'Localized bird droppings cause partial shading',
                'priority': 'Medium'
            },
            'snow_covered': {
                'expected_efficiency': 58.0,
                'std_deviation': 27.8,
                'range': (5, 95),
                'most_common': '90-100%',
                'description': 'Snow coverage blocks sunlight',
                'priority': 'Medium'
            },
            'physical_damage': {
                'expected_efficiency': 81.4,
                'std_deviation': 20.8,
                'range': (5, 95),
                'most_common': '90-100%',
                'description': 'Physical damage may have localized impact',
                'priority': 'High'
            },
            'electrical_damage': {
                'expected_efficiency': 73.6,
                'std_deviation': 22.1,
                'range': (15, 95),
                'most_common': '90-100%',
                'description': 'Electrical issues affect power conversion',
                'priority': 'High'
            }
        }

    def predict_efficiency(self, defect_type: str, confidence: float = 1.0) -> Dict:
        """
        Predict efficiency based on defect type and classification confidence.

        Args:
            defect_type (str): Detected defect type
            confidence (float): Classification confidence (0.0 to 1.0)

        Returns:
            Dict: Efficiency prediction with uncertainty bounds
        """
        if defect_type not in self.efficiency_models:
            return {
                'error': f'Unknown defect type: {defect_type}',
                'predicted_efficiency': 0,
                'confidence_interval': (0, 0),
                'maintenance_priority': 'Unknown'
            }

        model = self.efficiency_models[defect_type]

        # Base prediction
        base_efficiency = model['expected_efficiency']
        base_std = model['std_deviation']

        # Adjust uncertainty based on classification confidence
        # Lower confidence = higher uncertainty in efficiency prediction
        adjusted_std = base_std * (2.0 - confidence)

        # Calculate confidence interval
        lower_bound = max(0, base_efficiency - adjusted_std)
        upper_bound = min(100, base_efficiency + adjusted_std)

        # Calculate efficiency loss compared to clean panels
        clean_efficiency = self.efficiency_models['clean']['expected_efficiency']
        efficiency_loss = clean_efficiency - base_efficiency

        # NEW: Better loss display logic
        if efficiency_loss <= 0:
            # Panel performing better than average clean panel
            loss_display = f"+{abs(efficiency_loss):.1f}%"
            loss_description = "performing above average"
            loss_percent_display = "Better than average"
        else:
            # Panel performing worse than clean panel
            loss_display = f"{efficiency_loss:.1f}%"
            loss_description = "efficiency loss"
            loss_percent_display = f"{(efficiency_loss / clean_efficiency) * 100:.1f}% reduction"

        return {
            'predicted_efficiency': round(base_efficiency, 1),
            'confidence_interval': (round(lower_bound, 1), round(upper_bound, 1)),
            'std_deviation': round(adjusted_std, 1),
            'efficiency_loss': round(efficiency_loss, 1),
            'efficiency_loss_percent': round((efficiency_loss / clean_efficiency) * 100, 1),
            'maintenance_priority': model['priority'],
            'description': model['description'],
            'most_common_range': model['most_common'],
            'historical_range': model['range'],
            # NEW: Better display fields
            'loss_display': loss_display,
            'loss_description': loss_description,
            'loss_percent_display': loss_percent_display,
            'performance_status': 'Above Average' if efficiency_loss <= 0 else 'Below Average'
        }

    def get_recommendations(self, defect_type: str) -> list:
        """Get maintenance recommendations for a defect type."""
        recommendations = {
            'clean': [
                'Continue regular cleaning schedule',
                'Monitor for dust accumulation',
                'Check for minor obstructions'
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
                'Consider panel angle adjustments'
            ],
            'snow_covered': [
                'Monitor weather conditions',
                'Safe snow removal when appropriate',
                'Allow natural melting when possible',
                'Optimize panel tilt for snow shedding'
            ],
            'physical_damage': [
                'Professional inspection required',
                'Assess structural integrity',
                'Plan for component replacement',
                'Document damage for warranty claims'
            ],
            'electrical_damage': [
                'Professional electrical inspection required',
                'Check connections and wiring',
                'Test individual cell performance',
                'Verify inverter functionality',
                'Monitor for hot spots or arcing'
            ]
        }

        return recommendations.get(defect_type, ['Contact maintenance professional'])