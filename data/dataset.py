import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MediumMultimodalDataset(Dataset):
    """Loads paired Endoscopy (Kvasir v2) and Histopathology (LC25000) images."""
    def __init__(self, data_dir='/kaggle/input', transform=None, num_samples=4000):
        self.transform = transform
        endo_images = []
        histo_benign = []
        histo_malignant = []
        
        print("Scanning inputs for robust labels...")
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, file)
                    if 'kvasir' in root.lower():
                        endo_images.append(full_path)
                    elif 'colon_n' in root.lower(): 
                        histo_benign.append((full_path, 0))
                    elif 'colon_aca' in root.lower(): 
                        histo_malignant.append((full_path, 1))
        
        if not endo_images or not histo_benign or not histo_malignant:
            raise FileNotFoundError("Images missing. Ensure Kvasir and LC25000 datasets are attached.")
            
        print(f"Found {len(endo_images)} endoscopy images.")
        print(f"Found {len(histo_benign) + len(histo_malignant)} labeled histopathology images.")
        
        all_histo = histo_benign + histo_malignant
        random.shuffle(all_histo)
        random.shuffle(endo_images)
        
        self.samples = []
        self.num_samples = min(len(endo_images), len(all_histo), num_samples)
        
        for i in range(self.num_samples):
            e_path = endo_images[i]
            h_path, label = all_histo[i]
            self.samples.append((e_path, h_path, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        e_path, h_path, label = self.samples[idx]
        img_endo = Image.open(e_path).convert('RGB')
        img_histo = Image.open(h_path).convert('RGB')

        if self.transform:
            img_endo = self.transform(img_endo)
            img_histo = self.transform(img_histo)

        return img_endo, img_histo, label

def get_dataloaders(data_dir, batch_size=16, num_samples=4000, train_ratio=0.8):
    from torch.utils.data import random_split
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MediumMultimodalDataset(data_dir=data_dir, transform=transform, num_samples=num_samples)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader