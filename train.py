import os, json, random, logging
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageFile
from apihelper.models import Resnet50AutoEnc,ViTAutoEnc2
from apihelper.custom_datasets import CustomImageDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
save_path = './models/ViTAutoEnc3'
os.makedirs(save_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=f"{save_path}/app.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def log_metrics(model, dataset, device, epoch, history,save_path):
    model.eval()
    with torch.no_grad():
        idx = random.randint(0,len(os.listdir('./dataset')))
        img = Image.open('./dataset/'+os.listdir('./dataset')[idx]).convert('RGB')
        img = dataset.transform(img).unsqueeze(0).to(device)
        reconstructed_image = model(img)
    processed_image = vutils.make_grid(reconstructed_image.squeeze(0).detach().cpu(), padding=2, normalize=True)
    plt.imsave(f'{save_path}/output_image{epoch}.png', np.transpose(processed_image,(1,2,0)).numpy())    
    processed_image = vutils.make_grid(img.squeeze(0).detach().cpu(), padding=2, normalize=True)
    plt.imsave(f'{save_path}/input_image{epoch}.png', np.transpose(processed_image,(1,2,0)).numpy())    
    model.train()
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
    print(f"Model saved at {save_path}/model_checkpoint.pth")

if __name__ == '__main__':
    dataset = CustomImageDataset('./dataset')
    
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate the model
    model = ViTAutoEnc2()


    device = torch.device("mps")
    model.to(device)
    # Loss function
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.000005)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training loop
    num_epochs = 10
    # show loss
    log_steps = 50
    save_steps = 200
    history = []
    load_checkpoint = False
    start_epoch = -1
    if load_checkpoint:
        checkpoint = torch.load(f'{save_path}/model_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Model loaded from {save_path}/model_checkpoint.pth")
    model.train()
    for epoch in range(start_epoch+1,num_epochs):
        running_loss = 0.0
        for step,(inputs, _) in enumerate(tqdm(dataloader, total=len(dataloader))):
            inputs = inputs.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (step+1) % log_steps == 0:    # Print every log_loss mini-batches
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {running_loss/log_steps:.4f} lr: {lr}")
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {running_loss/log_steps:.4f} lr: {lr}")
                history.append(running_loss/log_steps)
                running_loss = 0.0
                with open(f'{save_path}/history.json','w') as f:
                    f.write(json.dumps(history))
            if (step+1) % save_steps == 0:
                save_model(model, optimizer, epoch, save_path)
        # scheduler.step()

        # Save the model, optimizer, and other relevant info
        save_model(model, optimizer, epoch,save_path)
        # eval with a random image from ./dataset
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        log_metrics(model, dataset, device, epoch, history,save_path)

    print('Finished Training')