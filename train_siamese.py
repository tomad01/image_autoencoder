import os, json, random, logging, sys, pdb
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from apihelper.models import SiamResNet,get_device
from apihelper.custom_datasets import CustomImageDataset2


save_path = './models/SiamResNet'
dataset_path = './pairs/pairs.json'
root_dir = '../dataset'
os.makedirs(save_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=f"{save_path}/app.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def log_metrics(epoch, history,save_path):
    # save loss plot
    plt.plot(history)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{save_path}/loss_plot.png')
    plt.close()
    logging.info(f"Metrics logged at epoch {epoch}")

def save_model(model, optimizer, epoch,save_path):
    # Save the model, optimizer, and other relevant info
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'{save_path}/model_checkpoint.pth')
    model.save_encoder(f'{save_path}/encoder_checkpoint.pth')
    print(f"Model saved at {save_path}/model_checkpoint.pth")

if __name__ == '__main__':
    dataset = CustomImageDataset2(dataset_path,root_dir)
    
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Instantiate the model
    device = get_device()
    model = SiamResNet(checkpoint_path='./models/encoder_checkpoint.pth',device=device)


    device = torch.device(device)
    model.to(device)


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training loop
    num_epochs = 3
    # show loss
    log_steps = 10
    save_steps = 100
    history = []
    start_epoch = -1

    model.train()

    for epoch in range(start_epoch+1,num_epochs):
        running_loss = 0.0
        for step,(input1, input2,label) in enumerate(tqdm(dataloader, total=len(dataloader))):
            input1 = input1.reshape((input1.shape[0],3,224,224)).to(device)
            input2 = input2.reshape((input2.shape[0],3,224,224)).to(device)
            label = label.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss,embeddings = model(input1,input2,label)
            
            # Backward pass and optimize
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (step+1) % log_steps == 0:    # Print every log_steps mini-batches
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {running_loss/log_steps:.4f} lr: {lr}")
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {running_loss/log_steps:.4f} lr: {lr}")
                embeddings = embeddings.detach().cpu().numpy()
                logging.info(f"Embeddings min value: {embeddings.min()}, max value: {embeddings.max()}")
                history.append(running_loss/log_steps)
                running_loss = 0.0
                with open(f'{save_path}/history.json','w') as f:
                    f.write(json.dumps(history))
                # break if embeddings are nan
                if np.isnan(embeddings).any():
                    print("Embeddings are nan. Exiting training")
                    sys.exit(1)
            if (step+1) % save_steps == 0:
                save_model(model, optimizer, epoch, save_path)
                log_metrics(epoch, history,save_path)
        scheduler.step()

        # Save the model, optimizer, and other relevant info
        save_model(model, optimizer, epoch,save_path)
        # eval with a random image from ./dataset
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        log_metrics(epoch, history,save_path)

    print('Finished Training')