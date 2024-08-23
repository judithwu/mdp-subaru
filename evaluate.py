import torch
from util import write_to_csv

def eval(epoch, model, test_dataloader, loss_fn, device, eval_filename):
        model.eval() #Set model to eval mode
        epoch_loss = 0
        epoch_accuracy = 0
    
        num_batches = len(test_dataloader)
        for images, label in test_dataloader:
                
                images = images.to(device=device)
                label = label.to(device=device)

                with torch.autocast(device):
                    pred = model(images) #Make predictions
                    output = torch.sigmoid(pred)
                    #convert probabilities to binary predictions
                    predictions = (output > 0.5).float()
                    pred = pred.squeeze()
                    loss = loss_fn(pred.float(), label.float()) #Crossentropy loss  

                    epoch_loss += loss.item() #Add loss to running loss total

                    correct = (predictions == label).float()
                    accuracy = correct.mean()
                    epoch_accuracy += accuracy.item()

        #After processing all batches, average the values to get the average metrics of the epoch
        epoch_loss = epoch_loss/num_batches
        epoch_accuracy = epoch_accuracy/num_batches
 
        #epoch_focal = epoch_focal/num_batches
        #epoch_boundary = epoch_boundary/num_batches
        #Write metrics to CSV file
        metrics = {"Epoch": epoch, 
                    "Loss": epoch_loss, 
                    "Accuracy": epoch_accuracy}
        filename = eval_filename
        write_to_csv(filename, metrics)
        model.train()
        return epoch_loss
                
      
   



