import torch.nn as nn
import torchvision.models as models
import torch

class ResNet101Embeddings(nn.Module):
    def __init__(self):
        super(ResNet101Embeddings, self).__init__()
        
        # Load pre-trained ResNet101 model
        resnet = models.resnet101(pretrained=True)
        
        # Remove the final fully connected layer and average pool layer
        # Keep the feature extractor part (up to the last conv block)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adding the adaptive pooling to make sure we can handle varying input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # This will pool the 7x7 feature map to 1x1
        
        # Flatten the pooled feature map into a 2048-dim vector
        self.flatten = nn.Flatten()
        self.nn  = nn.Linear(2048,1024)


    def forward(self, x):
        # Pass through ResNet101 encoder to get the feature map
        features = self.encoder(x)
        
        # Apply adaptive pooling to get a 1x1 feature map
        pooled_features = self.pool(features)
        
        # Flatten to obtain a (batch_size, 2048) embedding
        embeddings = self.flatten(pooled_features)
        
        return self.nn(embeddings)


class ResNet101Autoencoder(nn.Module):
    def __init__(self):
        super(ResNet101Autoencoder, self).__init__()
        
        self.encoder = ResNet101Embeddings()


        self.decoder = nn.Sequential(
            nn.Linear(1024, 768*14*14),
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

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 224x224 -> 448x448
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 3, kernel_size=1),  # Adjust channels
            nn.Tanh()             
         )

        count = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f'The decoder has {count/10**6:.2f} million trainable parameters')
        
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'The encoder has {count/10**6:.2f} million trainable parameters')
                # Define normalization parameters (mean and std for ImageNet)
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


    def forward(self, x, decode=True):
        # Encoder pass with ViT

        embeddings = self.encoder(x)  # Get the feature from ViT (batch_size, vit_embedding_dim)
        # embeddings = self.encoder(x)
        # print(embeddings.shape)
        # if torch.isnan(embeddings).any():
        #     print("NaN found in encoder output")

        # # embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)

        # if torch.isnan(embeddings).any():
        #     print("NaN found in normalized embeddings")
        # Decoder pass
        if decode:
            x = self.decoder(embeddings)  # Decoding
            # x_max = x.max(dim=[1, 2, 3], keepdim=True)[0]  # Max value per image, shape: (batch_size, 1, 1, 1)
            # x = x / x_max  # Divide the tensor by the maximum value
            # x = (x - self.mean) / self.std
        return x,embeddings
            
    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved to {path}")
        
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")
