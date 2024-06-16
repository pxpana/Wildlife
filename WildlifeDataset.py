import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WildlifeDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.data = pd.read_csv("metadata.csv")
        self.data = self.data[self.data['split'] == split]
        
        self.transform = transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx].path
        label = self.data.iloc[idx].identity
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label