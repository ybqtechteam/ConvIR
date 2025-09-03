import os
from PIL import Image as Image
from data import *
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from PIL import ImageFile
from pathlib import Path
import random




class ConcatDataset(Dataset):

    def __init__(self, datasets: dict, transform=None):
        self.datasets = datasets
        self.data = []

        for k, _ in self.datasets.items():
            for img in self.datasets[k]['images']:
                self.data.append(
                    {
                        'image': self.datasets[k]['dir'].joinpath('hazy', img),
                        'label': self.datasets[k]['dir'].joinpath('gt', img)
                    }
                )

        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]['image']).convert('RGB')
        label = Image.open(self.data[idx]['label']).convert('RGB')


        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        parts = self.data[idx]['image'].parts
        name = parts[-3] + '_' + parts[-1]
        
        return image, label, name



    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPG']:
                raise ValueError




class RandomSampler(Sampler):

    def __init__(self, total_samples: int, subset_ratio: float, shuffle: bool):
        self.total_samples = total_samples
        self.subset_samples = int(total_samples * subset_ratio)
        self.shuffle = shuffle

    def __iter__(self):
        indices = random.sample(list(range(self.total_samples)), self.subset_samples)

        if self.shuffle:
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.subset_samples
    



def test_subset_dataloader(subset_ratio, batch_size=1, num_workers=0):
    print('Test dataloader')

    datasets = {
            1 : {
                'dir' : Path('dataset/combined_test_set/Dense_Haze'),
                'images': sorted(os.listdir(Path('dataset/combined_test_set/Dense_Haze/hazy')), key=lambda x: int(x.split('.')[0])),
            },
            
            2: {
                'dir' : Path('dataset/combined_test_set/I-HAZE'),
                'images' : sorted(os.listdir(Path('dataset/combined_test_set/I-HAZE/hazy')), key=lambda x: int(x.split('.')[0])),
            },

            3: {
                'dir' : Path('dataset/combined_test_set/O-HAZY'),
                'images' : sorted(os.listdir(Path('dataset/combined_test_set/O-HAZY/hazy')), key=lambda x: int(x.split('.')[0])),
            },

            4: {
                'dir' : Path('dataset/combined_test_set/NH-HAZE'),
                'images' : sorted(os.listdir(Path('dataset/combined_test_set/NH-HAZE/hazy')), key=lambda x: int(x.split('.')[0])),
            },

            5: {
                'dir' : Path('dataset/combined_test_set/synthetic'),
                'images' : sorted(os.listdir(Path('dataset/combined_test_set/synthetic/hazy')), key=lambda x: int(x.split('.')[0])),
            }
        }
    

    transform = PairCompose(
            [
                PairResize((640, 480)),
                PairToTensor()
            ]
        )

    dataset = ConcatDataset(datasets, transform=transform)
    sampler = RandomSampler(len(dataset), subset_ratio=subset_ratio, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, sampler=sampler)
    return dataloader