import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ModifiedResNetEncoder, self).__init__()
        
        # Load ResNet50 pretrained model
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the ResNet to handle larger images by removing the last pooling layer
        # and reducing the stride in the first convolutional layer
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 512x512
            resnet.bn1,
            resnet.relu,
            # nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 256x256
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            resnet.maxpool,  # 256x256
            resnet.layer1,   # 256x256
            resnet.layer2,   # 128x128
            resnet.layer3,   # 64x64
            resnet.layer4,    # 32x32
            nn.Conv2d(
                in_channels=2048,
                out_channels=2048,
                kernel_size=3,
                stride=2,
                padding=1
            ),  # 16x16
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(
            in_channels=2048,
            out_channels=2048,
            kernel_size=3,  # Keep the kernel size as 3x3
            stride=2,       # Stride of 2 will reduce the spatial dimensions by half (16x16 -> 8x8)
            padding=1       # Padding of 1 will maintain the spatial dimensions after convolution
            ),  # 8x8
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1536)
        )
        count = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print(f"Encoder parameters: {count/10**6:.2f} milion")

    def forward(self, x):
        x = self.encoder(x)
        return x

# Define the Decoder as before
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(1536, 512 * 32 * 32),
            nn.ReLU(True),
            nn.Unflatten(1, torch.Size([512, 32, 32])),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # Upsample to 64x64
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),   # Upsample to 128x128
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # Upsample to 256x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # Upsample to 512x512
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # Upsample to 1024x1024
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),       # Final output layer
            nn.Tanh()  # Assuming the output needs to be in range [-1, 1]
        )
        count = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print(f"Decoder parameters: {count/10**6:.2f} milion")

    def forward(self, x):
        x = self.decoder(x)
        return x

# Define the complete Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = ModifiedResNetEncoder(pretrained=pretrained)
        self.decoder = Decoder()

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def forward(self, original_image):

        embedding = self.encoder(original_image)
        
        print(embedding.shape)
        reconstructed_image = self.decoder(embedding)

        return reconstructed_image


# Example usage
if __name__ == "__main__":
    model = EncoderDecoder(pretrained=True)
    
    # Create a dummy input tensor with a batch size of 1 and image size 1024x1024
    input_tensor = torch.randn(1, 3, 512, 512)
    
    # Pass the input tensor through the model
    output_tensor = model(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)