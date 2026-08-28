import os
import sys
import numpy as np

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfidenceGate:
    def __init__(self, min_confidence=0.65, min_margin=0.15):
        """
        Initializes the confidence gate with configurable thresholds.
        
        Args:
            min_confidence (float): Minimum acceptable probability for the top predicted class.
            min_margin (float): Minimum difference between top 1 and top 2 class probabilities.
        """
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def evaluate_predictions(self, predictions):
        """
        Evaluates predictions array to check confidence and margin.
        
        Args:
            predictions (np.ndarray): Softmax probabilities from the model, shape (num_classes,) or (1, num_classes).
            
        Returns:
            dict: Evaluation results containing uncertainty metrics.
        """
        # Ensure we have a 1D array of shape (num_classes,)
        preds = np.squeeze(np.array(predictions))
        
        if preds.ndim != 1 or len(preds) < 2:
            raise ValueError(f"Predictions must be a 1D array with at least 2 classes. Got shape: {preds.shape}")
            
        # Get sorted indices descending
        sorted_indices = np.argsort(preds)[::-1]
        top_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1])
        
        top_prob = float(preds[top_idx])
        second_prob = float(preds[second_idx])
        margin = float(top_prob - second_prob)
        
        # Check thresholds
        low_confidence = top_prob < self.min_confidence
        low_margin = margin < self.min_margin
        uncertain = low_confidence or low_margin
        
        reason = None
        if low_confidence:
            reason = "low_confidence"
        elif low_margin:
            reason = "low_margin"
            
        return {
            "passed": not uncertain,
            "uncertain": uncertain,
            "top_class_idx": top_idx,
            "top_confidence": top_prob,
            "second_confidence": second_prob,
            "margin": margin,
            "reason": reason
        }

if __name__ == "__main__":
    gate = ConfidenceGate()
    print("Testing confident prediction:")
    print(gate.evaluate_predictions([0.05, 0.80, 0.10, 0.05]))
    
    print("\nTesting low confidence prediction:")
    print(gate.evaluate_predictions([0.30, 0.40, 0.20, 0.10]))
    
    print("\nTesting low margin prediction:")
    print(gate.evaluate_predictions([0.02, 0.49, 0.47, 0.02]))
