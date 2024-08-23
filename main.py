from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
import gc

from dataset import DrowsyDataset
from model import ResNet50
from train import train


#paths to data folders
awake = r"D:\data\awake"
drowsy = r"D:\data\drowsy"

BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1
train_filename = "training_metrics_test.csv"
eval_filename = "eval_metrics_test.csv"

def main():
  #Create dataset
  dataset = DrowsyDataset(drowsy, awake)

  #Create dataloaders
  length_of_dataset = len(dataset)
  num_test = int(length_of_dataset * 0.2)
  num_train = length_of_dataset - num_test

  train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

  train_dataloader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
  val_dataloader = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
  
  #Initialize model, loss, optimizer
  model = ResNet50(NUM_CLASSES).to(DEVICE)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  #Reduce LR when loss stops improving for 3 epochs
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  torch.cuda.empty_cache()
  gc.collect()

  #Train model
  train(model, EPOCHS, train_dataloader, val_dataloader, scheduler, 
        optimizer, criterion, scaler, DEVICE, num_train, BATCH_SIZE, train_filename, eval_filename)

  #Save model
  save_name = "model_test.pth"
  state_dict = model.state_dict()
  torch.save(state_dict, save_name)

if __name__ == "__main__":
  main()