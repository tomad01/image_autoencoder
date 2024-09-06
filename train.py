import os, json, random, logging, sys, pdb
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageFile
from apihelper.models import Resnet50AutoEnc,ViTAutoEnc2,ResNet101Autoencoder
from apihelper.custom_datasets import CustomImageDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
save_path = './models/ResNet101Autoencoder'
dataset_path = '../dataset'
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
        idx = random.randint(0,len(os.listdir(dataset_path)))
        img = Image.open(os.path.join(dataset_path,os.listdir(dataset_path)[idx])).convert('RGB')
        img = dataset.transform(img).unsqueeze(0).to(device)
        reconstructed_image,_ = model(img)
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
    model.save_encoder(f'{save_path}/encoder_checkpoint.pth')
    print(f"Model saved at {save_path}/model_checkpoint.pth")

if __name__ == '__main__':
    dataset = CustomImageDataset(dataset_path)
    
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Instantiate the model
    model = ResNet101Autoencoder()


    device = torch.device("cuda")
    model.to(device)
    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training loop
    num_epochs = 10
    train_just_decoder_epochs = 1
    # show loss
    log_steps = 10
    save_steps = 100
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
    if train_just_decoder_epochs > 0:
        model.freeze_encoder()
    for epoch in range(start_epoch+1,num_epochs):
        if train_just_decoder_epochs > 0 and epoch == train_just_decoder_epochs:
            model.unfreeze_encoder()
            logging.info("Encoder unfrozen.")
        running_loss = 0.0
        for step,(inputs, _) in enumerate(tqdm(dataloader, total=len(dataloader))):
            inputs = inputs.to(device)
            dim0 = inputs.shape[0]
            inputs = inputs.reshape((dim0,3,224,224))
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs,embeddings = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, inputs)
            
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
                reconstructed_images = outputs.detach().cpu().numpy()
                embeddings = embeddings.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                logging.info(f"inputs min pixel value: {inputs.min()}, max pixel value: {inputs.max()}, mean pixel value: {inputs.mean()}")
                logging.info(f"reconstructed_images min pixel value: {reconstructed_images.min()}, max pixel value: {reconstructed_images.max()}")
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
                log_metrics(model, dataset, device, epoch, history,save_path)
        scheduler.step()

        # Save the model, optimizer, and other relevant info
        save_model(model, optimizer, epoch,save_path)
        # eval with a random image from ./dataset
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        log_metrics(model, dataset, device, epoch, history,save_path)

    print('Finished Training')