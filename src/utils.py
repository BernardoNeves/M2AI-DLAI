import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(targets, predictions, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(train_losses, val_metrics, save_path):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [m['accuracy'] for m in val_metrics], 'g-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs, [m['f1'] for m in val_metrics], 'm-', label='Validation F1')
    ax3.set_title('F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(epochs, [m['auc_roc'] for m in val_metrics], 'c-', label='Validation AUC-ROC')
    ax4.set_title('AUC-ROC Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC-ROC')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_metrics': val_metrics,
        'model_name': model.model_name
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
