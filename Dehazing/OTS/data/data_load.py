import os, math
import torch, random
import numpy as np
from PIL import Image as Image
from data import *
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')
    print('Train dataloader')

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
        pin_memory=True,
        drop_last=True  
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

    transform = PairCompose(
            [
                PairResize((640, 480)),
                PairToTensor()
            ]
        )
    
    dataloader = DataLoader(
        Dataset(os.path.join(path, 'val'), transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class Dataset(TorchDataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx])).convert('RGB')

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









class CalibrationDataset(TorchDataset):
    """
    Dataset per la calibrazione della quantizzazione.
    Carica solo le immagini di input (senza etichette).
    """
    def __init__(self, image_dir, transform=None):
        # directory contenente le immagini da usare per la calibrazione
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image

    @staticmethod
    def _check_image(lst):
        valid_ext = {'png', 'jpg', 'jpeg', 'JPG'}
        for x in lst:
            ext = x.split('.')[-1]
            if ext not in valid_ext:
                raise ValueError(f"Formato file non supportato: {x}")




'''
-----------------------------------------------------------------------------------------------------------------------
'''







import pathlib
import random
class CurriculumLearningDataset(Dataset):

    def __init__(self, easy_dataset, medium_dataset, hard_dataset, extreme_dataset, transform=None, is_test=False):

        self.datasets = {
            'easy': {
                'image_dir' : easy_dataset,
                'image_list' : os.listdir(os.path.join(easy_dataset, 'hazy')),
            },
            'medium': {
                'image_dir' : medium_dataset,
                'image_list' : os.listdir(os.path.join(medium_dataset, 'hazy/')),
            },
            'hard': {
                'image_dir' : hard_dataset,
                'image_list' : os.listdir(os.path.join(hard_dataset, 'hazy/')),
            },
            'extreme': {
                'image_dir' : extreme_dataset,
                'image_list' : os.listdir(os.path.join(extreme_dataset, 'hazy/'))
            }
        }

        

        self.data = []
        for k, _ in self.datasets.items():
            self.datasets[k]['image_list'].sort()
            for file in self.datasets[k]['image_list']:
                self.data.append({'image' : os.path.join(self.datasets[k]['image_dir'], 'hazy', file),
                                  'label' : os.path.join(self.datasets[k]['image_dir'], 'gt', file),
                                  'difficulty' : k})

        self.transform = transform
        self.is_test = is_test
    
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

        if self.is_test:
            name = os.path.basename(self.data[idx]['image'])
            return image, label, name
        
        return image, label, self.data[idx]['image']



    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPG']:
                raise ValueError
            


    def length(self, difficulty):
        return len([x for x in self.data if x['difficulty'] == difficulty])
    

    def get_difficulty_indicies(self):
        easy_indices = self._get_indices('easy')
        medium_indices = self._get_indices('medium')
        hard_indices = self._get_indices('hard')
        extreme_indices = self._get_indices('extreme')
        return easy_indices, medium_indices, hard_indices, extreme_indices
    
    def _get_indices(self, difficulty):
        return [i for i, x in enumerate(self.data) if x['difficulty'] == difficulty]








class CurriculumSampler(Sampler):
    def __init__(self, indicies, phase, dataset_length, shuffle): # dataset_length is the total number of samples of the current dataset
        self.phase = phase
        self.easy_indices, self.medium_indices, self.hard_indices, self.extreme_indices = indicies
        self.dataset_length = dataset_length
        self.shuffle = shuffle

    def __iter__(self):
        
        if self.phase == "medium":
            n_medium = self.dataset_length
            x = self.dataset_length * 0.20
            n_easy = math.ceil(x) if x <= 0.5 else int(x)

            sampled_medium = random.sample(self.medium_indices, n_medium)
            sampled_easy = random.sample(self.easy_indices, n_easy)

            indices = sampled_medium + sampled_easy

        elif self.phase == "hard":
            n_hard = self.dataset_length
            x = self.dataset_length * 0.10
            n_medium = n_easy = math.ceil(x) if x <= 0.5 else int(x)

            sampled_hard = random.sample(self.hard_indices, n_hard)
            sampled_medium = random.sample(self.medium_indices, n_medium)
            sampled_easy = random.sample(self.easy_indices, n_easy)

            indices = sampled_hard + sampled_medium + sampled_easy
        
        elif self.phase == "extreme":
            n_extreme = self.dataset_length
            x = self.dataset_length * 0.10
            n_hard = n_medium = n_easy = math.ceil(x) if x <= 0.5 else int(x)

            sampled_extreme = random.sample(self.extreme_indices, n_extreme)
            sampled_hard = random.sample(self.hard_indices, n_hard)
            sampled_medium = random.sample(self.medium_indices, n_medium)
            sampled_easy = random.sample(self.easy_indices, n_easy)

            indices = sampled_extreme + sampled_hard + sampled_medium + sampled_easy

        else:
            raise ValueError("Phase must be either 'medium' 'or 'hard' or 'extreme'")

        if self.shuffle:
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.dataset_length
    



    
def train_dataloader_CL(folders, phase, batch_size=64, num_workers=0, use_transform=True):
    print('Train dataloader')

    easy_dataset = pathlib.Path(folders['easy']) / 'train'
    medium_dataset = pathlib.Path(folders['medium']) / 'train'
    hard_dataset = pathlib.Path(folders['hard']) / 'train'
    extreme_dataset = pathlib.Path(folders['extreme']) / 'train'


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


    dataset = CurriculumLearningDataset(easy_dataset, medium_dataset, hard_dataset, extreme_dataset, transform=transform) 
    sampler = CurriculumSampler(dataset.get_difficulty_indicies(), phase=phase, dataset_length=dataset.length(phase), shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler, drop_last=True) # ultimo batch se non completo da errore
    return dataloader
    






def valid_dataloader_CL(folders, phase, batch_size=64, num_workers=0, use_transform=True):
    print('Valid dataloader')

    easy_dataset = pathlib.Path(folders['easy']) / 'val'
    medium_dataset = pathlib.Path(folders['medium']) / 'val'
    hard_dataset = pathlib.Path(folders['hard']) / 'val'
    extreme_dataset = pathlib.Path(folders['extreme']) / 'val'


    transform = None
    if use_transform:
        transform = PairCompose(
            [   
                PairResize((640, 480)),
                PairToTensor()
            ]
        )


    dataset = CurriculumLearningDataset(easy_dataset, medium_dataset, hard_dataset, extreme_dataset, transform=transform) 
    sampler = CurriculumSampler(dataset.get_difficulty_indicies(), phase=phase, dataset_length=dataset.length(phase), shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return dataloader