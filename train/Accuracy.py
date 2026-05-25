import torch

class Accuracy:
    def __init__(self, threshold=0.5, ignore_index=-1, device=None):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, predictions, targets):
        predictions = predictions.squeeze().to(self.device)
        targets = targets.to(self.device)

        predicted_classes = (predictions > self.threshold).long()
        
        mask = targets != self.ignore_index
        predicted_classes = predicted_classes[mask]
        valid_targets = targets[mask]
        
        correct_predictions = (predicted_classes == valid_targets).sum().item()

        total_valid_samples = valid_targets.size(0)

        if total_valid_samples == 0:
            return 0.0

        return correct_predictions / total_valid_samples

