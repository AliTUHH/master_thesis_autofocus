# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(train_losses, test_losses, save_path=None):
    """Plot training and test MSE histories and optionally save the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (cm²)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(true_vals, pred_vals, save_path=None):
    """Compare predicted and true defocus values against the ideal diagonal."""
    plt.figure(figsize=(8, 8))
    plt.scatter(true_vals, pred_vals, alpha=0.5)
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('True Deviation (cm)')
    plt.ylabel('Predicted Deviation (cm)')
    plt.title('True vs Predicted Deviation')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
