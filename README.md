# Wildlife

## Download and Unzip the Data

First, download the images from Google Drive https://drive.google.com/file/d/1JKdwNPaRaviFvbJFEm4zBkf_dJdYxFK9/view?usp=sharing and unzip the folder.

## Using the WildlifeDataset Class

The `WildlifeDataset` class inherits from `torch.utils.data.Dataset`. You need to specify whether you want to use the train or test data and the transform function.

```python
from WildlifeDataset import WildlifeDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the transform function
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),            
])

# Initialize the dataset
dataset = WildlifeDataset(split="test", transform=transform)

# Apply it to the torch DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through the DataLoader
for images, labels in loader:
    pass
