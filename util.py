import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F

def plot_metrics(train_csv, eval_csv):
    # Load the CSV files
    train_data = pd.read_csv(train_csv)
    eval_data = pd.read_csv(eval_csv)

    # Create plots for Loss and Accuracy
    fig, ax= plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["Loss", "Accuracy"]

    for i, met in enumerate(metrics):
      ax[i].plot(train_data["Epoch"], train_data[met], label="Training", color="blue")
      ax[i].plot(eval_data["Epoch"], eval_data[met], label="Evaluation", color="red")
      ax[i].set_xlabel("Epoch")
      ax[i].set_ylabel(met)
      ax[i].set_title(f"{met} per Epoch")
      ax[i].legend()


    # Show the plots
    plt.tight_layout()
    plt.show()


def write_to_csv(filename, metrics):
    # Check if the file already exists to determine if need a header
    try:
        with open(filename, 'x', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
    except FileExistsError:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            writer.writerow(metrics)

def pad_img(image, target_height, target_width, mode='constant', value=0):
    _, height, width = image.shape
    
    # Calculate padding
    pad_height = target_height - height
    pad_width = target_width - width
    
    # Padding to be added on each side (left, right, top, bottom)
    padding = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)
    
    # Pad the image
    padded_image = F.pad(image, padding, mode=mode, value=value)
    
    return padded_image