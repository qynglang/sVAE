import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import PIL
import numpy as np
from typing import Any, Callable, List, Optional, Union, Tuple
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    def _check_integrity(self) -> bool:
        return True
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

#         target: Any = []
#         for t in self.target_type:
#             if t == "attr":
#                 target.append(self.attr[index, :])
#             elif t == "identity":
#                 target.append(self.identity[index, 0])
#             elif t == "bbox":
#                 target.append(self.bbox[index, :])
#             elif t == "landmarks":
#                 target.append(self.landmarks_align[index, :])
#             else:
#                 # TODO: refactor with utils.verify_str_arg
#                 raise ValueError(f'Target type "{t}" is not recognized.')
        if os.path.exists(self.root+ "/att/"+ "att1/"+str(int(self.filename[index][:-4]))+".npy"):
            target=np.load(self.root+ "/att/"+ "att1/"+str(int(self.filename[index][:-4]))+".npy")
            #print('1')
            target[48:88]*=2
            #target=target[48:98]
        else:
            target=np.zeros(98)
        #target[target==-1]=0

        if self.transform is not None:
            X = self.transform(X)

#         if target:
#             #target = tuple(target) if len(target) > 1 else target[0]

        if self.target_transform is not None:
            target = self.target_transform(target)
#         else:
#             target = None

        return X, target
    
    

class LFW(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face3/Face')       
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
        
        #self.label_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face3/att/att')
        #labels=sorted([f for f in self.label_dir.iterdir() if f.suffix == '.npy'])
        
        #self.labels = labels[:int(len(labels) * 0.75)] if split == "train" else labels[int(len(labels) * 0.75):]
        
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        if os.path.exists('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face3/att/att1/'+str(self.imgs[idx])[61:-3]+'npy'):
            label=np.load('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face3/att/att1/'+str(self.imgs[idx])[61:-3]+'npy')
            #label=np.load('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face3/att/att1/'+str(self.imgs[idx])[61:-3]+'npy')##121
            #label[48:121]*=5
            #label=np.hstack((label[0:10],label[24:34],label[48:121]))
        else:
            label=np.zeros(121)
        
        return img, label # dummy datat to prevent breaking 
    
class Face3D(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        #self.data_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face2/img2/img3c') 
        self.data_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face2/imgs') 
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.npy'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
        
        #self.label_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face2/att2/att2')
        self.label_dir = Path('/nfs/h1/userhome/wdh-llx/workingdir/zsk/sVAE/Data/Face2/att3/attb')
        labels=sorted([f for f in self.label_dir.iterdir() if f.suffix == '.npy'])
        
        self.labels = labels[:int(len(labels) * 0.75)] if split == "train" else labels[int(len(labels) * 0.75):]
        
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = np.load(self.imgs[idx])
        img=PIL.Image.fromarray(img)
        
        if self.transforms is not None:
            img = self.transforms(img)
        label=np.load(self.labels[idx])
        
        return img, label # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = Face3D(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = Face3D(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     