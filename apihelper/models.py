import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from timm import create_model
import torch

class ViTAutoEnc2(nn.Module):
    def __init__(self, vit_model='vit_base_patch16_224'):
        super(ViTAutoEnc2, self).__init__()
        
        # Use a pre-trained Vision Transformer as the encoder
        self.encoder = create_model(vit_model, pretrained=True)
        self.encoder.head = nn.Identity()
        # Extract the feature dimension from the ViT model
        self.vit_embedding_dim = self.encoder.embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(768, 768*14*14),
            nn.Unflatten(1, torch.Size([768, 14, 14])),
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=1),  # Adjust channels
            nn.Sigmoid()  # Normalize output to [0, 1]   
         )

        count = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f'The decoder has {count/10**6:.2f} million trainable parameters')
        
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'The encoder has {count/10**6:.2f} million trainable parameters')


    def forward(self, x, decode=True):
        # Encoder pass with ViT

        # x = self.encoder.forward_features(x).mean(dim=1)  # Get the feature from ViT (batch_size, vit_embedding_dim)
        x = self.encoder(x)

        # Decoder pass
        if decode:
            return self.decoder(x)  # Decoding
        else:
            return F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension

class ViTAutoEnc(nn.Module):
    def __init__(self, vit_model='vit_base_patch16_224'):
        super(ViTAutoEnc, self).__init__()
        
        # Use a pre-trained Vision Transformer as the encoder
        self.encoder = create_model(vit_model, pretrained=True)
        self.encoder.head = nn.Identity()
        # Extract the feature dimension from the ViT model
        self.vit_embedding_dim = self.encoder.embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(768, 768*7*7),
            nn.Unflatten(1, torch.Size([768, 7, 7])),
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=1),  # Adjust channels
            nn.Sigmoid()  # Normalize output to [0, 1]   
         )

        count = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f'The decoder has {count/10**6:.2f} million trainable parameters')
        
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'The encoder has {count/10**6:.2f} million trainable parameters')


    def forward(self, x, decode=True):
        # Decoder pass
        if decode:
            # x = self.encoder.forward_features(x).mean(dim=1)  # Get the feature from ViT (batch_size, vit_embedding_dim)
            x = self.encoder(x)
            return self.decoder(x)  # Decoding
        else:
            x = self.encoder(x) # Get the feature from ViT (batch_size, vit_embedding_dim)
            return F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension

class Resnet50AutoEnc(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Resnet50AutoEnc, self).__init__()
        
        # Use pre-trained ResNet50 as the encoder
        resnet = models.resnet50(pretrained=True)
        
        # Extract layers up to the layer 4 (which gives 2048 channels)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Reduce embedding size to the specified dimension (512 or 768)
        self.embedding_layer = nn.Conv2d(2048, embedding_dim, kernel_size=1)
        
        # To map the embedding vector to a form suitable for the decoder
        self.scale_down = nn.Linear(embedding_dim * 7 * 7, embedding_dim)

        self.scale_up = nn.Linear(embedding_dim, embedding_dim * 7 * 7)
        
        # Define the enriched decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 512, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=1),  # Adjust channels
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        count = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f'The decoder has {count/10**6:.2f} million trainable parameters')
        
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'The encoder has {count/10**6:.2f} million trainable parameters')
        count = sum(p.numel() for p in self.scale_down.parameters() if p.requires_grad)
        print(f'The scale_down has {count/10**6:.2f} million trainable parameters')
        count = sum(p.numel() for p in self.scale_up.parameters() if p.requires_grad)
        print(f'The scale_up has {count/10**6:.2f} million trainable parameters')
        count = sum(p.numel() for p in self.embedding_layer.parameters() if p.requires_grad)
        print(f'The embedding_layer has {count/10**6:.2f} million trainable parameters')

    def forward(self, x,decode=True):
        # Encoder pass
        x = self.encoder(x)  # Encoding to get 2048 features with spatial size 7x7
        x = self.embedding_layer(x)  # Reduce to embedding_dim features with spatial size 7x7
        
        # Flatten and fully connect to prepare for decoder
        x = x.view(x.size(0), -1)  # Flatten
        x = self.scale_down(x)  # Fully connected layer
        x = F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension

        # Decoder pass
        if decode:
            x = self.scale_up(x)
            x = x.view(x.size(0), -1, 7, 7)  # Reshape back to the spatial dimensions required by the decoder
            x = self.decoder(x)  # Decoding
        return x