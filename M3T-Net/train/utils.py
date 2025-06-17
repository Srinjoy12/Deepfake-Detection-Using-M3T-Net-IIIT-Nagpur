import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# Calculate accuracy
def compute_accuracy(y_true, y_pred):
    y_pred_label = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_label)

# Calculate AUC
def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return float('nan')

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

# Load model checkpoint
def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch
