import os
from PIL import Image as Image
from data import *
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, data='CSD', use_transform=True):
    print('Train dataloader')
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairResize((640, 480)),
                PairRandomHorizontalFlip(),
                PairRandomVerticalFlip(),
                PairRandomRotation(degrees=30),
                PairToTensor()
            ]
        )
    
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, data, batch_size=1, num_workers=0):
    print('Test dataloader')
    image_dir = os.path.join(path, 'test')

    transform = PairCompose(
        [
            PairResize((640, 480)),
            PairToTensor()
        ]
    )


    dataloader = DataLoader(
        DeblurDataset(image_dir, data, transform=transform, is_test=True),
        
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0):
    print('Valid dataloader')
    
    transform = PairCompose(
        [   
            PairResize((640, 480)),
            PairToTensor()
        ]
    )
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'val'), data, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'Snow/'))
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.data = data
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'Snow', self.image_list[idx]))
        if self.data == 'SRRS':
            label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))
        else:
            label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

