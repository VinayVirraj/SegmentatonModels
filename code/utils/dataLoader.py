import torch
from torch.utils.data import Dataset, DataLoader
from utils.imageUtils import load_images, data_split


class CottonDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, mask
    

def load_data(data_dir, transform, batch_size, train_ratio=0.7, val_ratio=0.2):
    images = load_images(f"{data_dir}/images")
    masks = load_images(f"{data_dir}/masks", is_mask=True)
    dataset = CottonDataset(images, masks, transform=transform)
    train_dataset, val_dataset, test_dataset = data_split(dataset, train_ratio, val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader