import torch
from tqdm import tqdm
from util import write_to_csv
from evaluate import eval


def train(model, epochs, train_dataloader, test_dataloader, scheduler, 
          optimizer, loss_fn, scaler, device, num_train, batch_size, train_filename, eval_filename):
    for epoch in range(1, epochs+1):
        
        model.train() #Set model to training mode
        epoch_loss = 0
        epoch_accuracy = 0
    
        num_batches = len(train_dataloader)
        with tqdm(total=num_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as progress_bar:
            for images, label in train_dataloader:
                
                images = images.to(device=device)
                label = label.to(device=device)

                with torch.autocast(device):
                    pred = model(images) #Make predictions
                    output = torch.sigmoid(pred)
                    #convert probabilities to binary predictions
                    predictions = (output > 0.5).float()
                    pred = pred.squeeze()
                    #print(pred)
                    #print(label)
                    loss = loss_fn(pred.float(), label.float()) #Crossentropy loss  

                    epoch_loss += loss.item() #Add loss to running loss total

                    correct = (predictions == label).float()
                    accuracy = correct.mean()
                    epoch_accuracy += accuracy.item()
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) #Unscale so optimizer can correctly apply updates to model params
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Clip gradients to prevent exploding gradient
                scaler.step(optimizer)
                scaler.update()

                progress_bar.update(batch_size)
                progress_bar.set_postfix(**{"Loss": loss.item(), 
                                            "Accuracy": accuracy.item()})
        #After processing all batches, average the values to get the average metrics of the epoch
        #print(vec)
        epoch_loss = epoch_loss/num_batches
        epoch_accuracy = epoch_accuracy/num_batches

        #epoch_focal = epoch_focal/num_batches
        #epoch_boundary = epoch_boundary/num_batches
        #Write metrics to CSV file
        metrics = {"Epoch": epoch, 
                    "Loss": epoch_loss, 
                    "Accuracy": epoch_accuracy}
        filename = train_filename
        write_to_csv(filename, metrics)

       
        #Evaluation round
        model.eval()
        print("Evaluation round")
        #with tqdm(total=num_test, desc=f"Epoch {epoch}", unit="img") as progress_bar:
        loss_score = eval(epoch, model, test_dataloader, loss_fn, device, eval_filename)
        scheduler.step(loss_score)
                
      
   



