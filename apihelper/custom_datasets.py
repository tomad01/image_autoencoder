# Define a custom Dataset class for image reconstruction
import os
from torchvision.transforms import transforms
from PIL import Image,ImageFile
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        # self.images = [self.transform(Image.open(os.path.join(root_dir, img)).convert('RGB')) for img in tqdm(os.listdir(root_dir))]

    def __len__(self):
        return len(self.image_paths)

    # def transform(self,image):
    #     return self.processor(images=image, return_tensors="pt")['pixel_values']

    def __getitem__(self, idx):
        # image = self.images[idx]
        image = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        return image,image