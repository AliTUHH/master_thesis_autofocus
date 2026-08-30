# -*- coding: utf-8 -*-
import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.generator import generate_dataset
from src.models.cnn import AutofocusCNN
from src.utils.metrics import calculate_mae
from src.utils.plotting import plot_loss_curves, plot_predictions

def main(config_path):
    """Run data generation, model training, evaluation, and artifact export.

    Parameters
    ----------
    config_path : str
        Path to the YAML file that defines the dataset, neural network,
        optimization, and output-directory settings.
    """

    # 1. Load the experiment configuration. YAML keeps the most frequently
    # changed hyperparameters outside the source code so experiments remain
    # reproducible and can be compared without editing this module.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🚀 بدء التدريب باستخدام الإعدادات:")
    print(f"   - حجم الصورة: {config['image_size']}")
    print(f"   - عدد العينات: {config['num_samples']}")
    print(f"   - عدد التكرارات (Epochs): {config['epochs']}")

    # 2. Generate synthetic holograms and their signed defocus labels. PyTorch
    # convolutions expect input shaped as (batch, channels, height, width), so a
    # singleton channel dimension is inserted for the grayscale intensities.
    X_raw, y_raw = generate_dataset(config)
    X_raw = X_raw[:, np.newaxis, :, :]  # Add the grayscale channel dimension.

    # 3. Split the arrays into training and test partitions. The generator
    # already randomizes the physical parameters, so a contiguous split still
    # contains independently sampled examples in both partitions.
    split = int((1 - config['test_split']) * len(X_raw))
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    # 4. Wrap NumPy arrays as tensor datasets and data loaders. Training samples
    # are shuffled once per epoch to reduce ordering effects; test samples are
    # kept in a stable order to make evaluation and plots reproducible.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=config['batch_size'], shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=config['batch_size'], shuffle=False
    )

    # 5. Construct the convolutional regression model. Adam updates the network
    # parameters, and mean squared error penalizes larger distance errors more
    # strongly than smaller ones.
    model = AutofocusCNN(config)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    # 6. Train for the configured number of epochs while recording one average
    # loss for each complete pass over the training and test partitions.
    train_losses = []
    test_losses = []
    print("\n🏋️ بدء التدريب...")

    for epoch in range(config['epochs']):
        # Training phase: enable BatchNorm updates and dropout, clear gradients
        # before every batch, backpropagate the MSE, and update all parameters.
        model.train()
        tr_loss = 0.0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        avg_tr_loss = tr_loss / len(train_loader)
        train_losses.append(avg_tr_loss)

        # Evaluation phase: disable dropout and stop BatchNorm statistics from
        # changing. Gradients are disabled to reduce memory and computation.
        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds = model(imgs)
                te_loss += criterion(preds, lbls).item()
        avg_te_loss = te_loss / len(test_loader)
        test_losses.append(avg_te_loss)

        # Report progress every ten epochs without flooding the terminal.
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_tr_loss:.6f} | Test Loss: {avg_te_loss:.6f}")

    print("✅ انتهى التدريب!")

    # 7. Collect final test-set predictions. The arrays are retained in
    # centimeters because this is the regression target used during training.
    model.eval()
    predictions_cm = []
    true_cm = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            preds = model(imgs)
            predictions_cm.extend(preds.numpy())
            true_cm.extend(lbls.numpy())

    predictions_cm = np.array(predictions_cm)
    true_cm = np.array(true_cm)

    # Convert signed centimeter deviations back to absolute distances in meters
    # before calculating the human-readable final error in millimeters.
    pred_m = predictions_cm / 100 + 1.0
    true_m = true_cm / 100 + 1.0

    mae_mm = calculate_mae(pred_m, true_m) * 1000
    print(f"\n📊 متوسط الخطأ المطلق (MAE): {mae_mm:.2f} مم")

    # 8. Create the output directory if necessary and save diagnostic figures:
    # the optimization history and a prediction-versus-reference scatter plot.
    os.makedirs(config['figures_dir'], exist_ok=True)
    plot_loss_curves(train_losses, test_losses, save_path=f"{config['figures_dir']}/loss_curve.png")
    plot_predictions(true_cm, predictions_cm, save_path=f"{config['figures_dir']}/scatter_plot.png")

    # 9. Save only the learned parameter state. Reconstructing the model later
    # requires the same architecture values from the YAML configuration.
    os.makedirs(config['model_dir'], exist_ok=True)
    torch.save(model.state_dict(), f"{config['model_dir']}/model_final.pth")
    print(f"💾 تم حفظ النموذج في: {config['model_dir']}/model_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
