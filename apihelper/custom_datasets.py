# Define a custom Dataset class for image reconstruction
import os, json
import torch
from torchvision.transforms import transforms
from PIL import Image,ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset2(Dataset):
    def __init__(self, data_path):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.pozitive_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # add noise to the image
            transforms.Lambda(lambda x: torch.clamp(x + 0.1 * torch.randn_like(x), min=0, max=1))

        ])
        with open(data_path) as f:
            dataset = json.load(f)
        self.parsed_dataset = dataset+dataset

    def __len__(self):
        return len(self.parsed_dataset)

    def __getitem__(self, idx):
        pair = self.parsed_dataset[idx]
        if idx< len(self.parsed_dataset)//2:
            image1 = self.transform(Image.open(pair['image1']).convert('RGB'))
            image2 = self.transform(Image.open(pair['image2']).convert('RGB'))
            similarity_score = torch.tensor(pair['similarity_score'])
        else:
            image1 = self.pozitive_transform(Image.open(pair['image1']).convert('RGB'))
            image2 = self.transform(Image.open(pair['image1']).convert('RGB'))
            similarity_score = torch.tensor(1)
        return image1,image2,similarity_score

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