# Define a custom Dataset class for image reconstruction
import os, json, random
import torch
from torchvision.transforms import transforms
from PIL import Image,ImageFile
from torch.utils.data import Dataset

def modify_random_region(img, max_region_size=0.5):
    """
    Modifies a random region of the image by adding noise or random values.
    
    Parameters:
        img (Tensor): The input image tensor (C, H, W).
        max_region_size (float): Maximum fraction of the image size to modify (0 < max_region_size <= 1).
    Returns:
        Tensor: The image with a random region modified.
    """
    _, h, w = img.shape
    
    # Randomly determine the size of the region (random height and width)
    region_h = random.randint(1, int(h * max_region_size))
    region_w = random.randint(1, int(w * max_region_size))
    
    # Choose a random starting point for the region, ensuring it stays within bounds
    top = random.randint(0, h - region_h)
    left = random.randint(0, w - region_w)
    
    # Modify the selected region (e.g., by adding noise or random values)
    img[:, top:top+region_h, left:left+region_w] += 0.8 * torch.randn_like(img[:, top:top+region_h, left:left+region_w])
    
    # Optionally, you can clamp the values to ensure they are within a valid range
    img = torch.clamp(img, min=0, max=1)
    
    return img

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset2(Dataset):
    def __init__(self, data_path,root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.pozitive_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Lambda(lambda x: torch.clamp(x + 0.1 * torch.randn_like(x), min=0, max=1)), # Add noise
            # transforms.Lambda(lambda x: modify_random_region(x, max_region_size=0.4)),  # Modify a random region
            transforms.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.2, 0.2)  # Max translation: 20% of image height and width
            ),
        ])
        self.pozitive_transform1 = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired size
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Lambda(lambda x: torch.clamp(x + 0.1 * torch.randn_like(x), min=0, max=1)), # Add noise
            transforms.Lambda(lambda x: modify_random_region(x, max_region_size=0.4)),  # Modify a random region
            transforms.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.2, 0.2)  # Max translation: 20% of image height and width
            ),
        ])
        with open(data_path) as f:
            dataset = json.load(f)
        self.parsed_dataset = dataset+dataset
        print(f"Dataset size: {len(self.parsed_dataset)}")

    def flip_coin(self):
        return bool(random.randint(0,1))

    def __len__(self):
        return len(self.parsed_dataset)

    def __getitem__(self, idx):
        pair = self.parsed_dataset[idx]
        image1 = Image.open(os.path.join(self.root_dir,pair['image1'])).convert('RGB')
        if idx< len(self.parsed_dataset)//2:
            image2 = Image.open(os.path.join(self.root_dir,pair['image2'])).convert('RGB')
            t_image1 = self.transform(image1)
            t_image2 = self.transform(image2)
            similarity_score = torch.tensor(pair['similarity_score'])
        else:
            
            if self.flip_coin():
                if self.flip_coin():
                    t_image1 = self.pozitive_transform(image1)
                    t_image2 = self.transform(image1)
                else:
                    t_image1 = self.pozitive_transform1(image1)
                    t_image2 = self.transform(image1)
            else:
                image2 = Image.open(os.path.join(self.root_dir,pair['image2'])).convert('RGB')
                if self.flip_coin():
                    t_image1 = self.pozitive_transform(image2)
                    t_image2 = self.transform(image2)
                else:
                    t_image1 = self.pozitive_transform1(image2)
                    t_image2 = self.transform(image2)
            similarity_score = torch.tensor(1)
        return t_image1,t_image2,similarity_score

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