import os
from PIL import Image as Image
from data import *
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
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
        Dataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    print('Test dataloader')

    image_dir = os.path.join(path, 'test')

    transform = PairCompose(
        [
            PairResize((640, 480)),
            PairToTensor()
        ]
    )


    dataloader = DataLoader(
        Dataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    print('Valid dataloader')
    
    path = os.path.join(path, 'val')
    
    transform = PairCompose(
        [   
            PairResize((640, 480)),
            PairToTensor()
        ]
    )
    dataloader = DataLoader(
        Dataset(path, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class Dataset(TorchDataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx])).convert('RGB')
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx])).convert('RGB')

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        return image, label, self.image_list[idx]

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPG']:
                raise ValueError
