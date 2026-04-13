import os
import cv2
import glob
import numpy as np
import albumentations as A
import PIL
from PIL import Image
from torch.utils.data import Dataset

class SurgicalDataset(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=128,
                 augment=False,
                 interpolation="lanczos",
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        image_path_ = []
        for path, root in zip(self.data_paths, self.data_root):
            with open(path, "r") as f:
                self.image_paths = f.read().splitlines()
                image_path_ += [item for sublist in [glob.glob(os.path.join(root, l, "*.jpg")) for l in self.image_paths] for item in sublist]

        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        
        if self.augment:
            self.rescaler = A.Compose([A.OneOf([
                                    A.Resize(height=self.size, width=self.size, interpolation=self.interpolation, p=0.7),
                                    A.RandomCrop(height=self.size, width=self.size, p=0.1),
                                    A.Compose([A.Resize(height=int(self.size * 1.50), width=int(self.size * 1.50), interpolation=self.interpolation), A.RandomCrop(height=self.size, width=self.size)], p=0.1),
                                    A.Compose([A.Resize(height=int(self.size * 1.25), width=int(self.size * 1.25), interpolation=self.interpolation), A.RandomCrop(height=self.size, width=self.size)], p=0.1),
                                    ], p=1.0)])

            self.augmentation = A.Compose([
                                A.HorizontalFlip(p=0.20),
                                A.VerticalFlip(p=0.10),
                                A.RandomRotate90(p=0.10),
                                A.RandomBrightnessContrast(p=0.20),
                                A.RandomGamma(p=0.20)
                                ])
        else:
            self.rescaler = A.Resize(height=self.size, width=self.size, interpolation=self.interpolation)

        self.labels = {
            "image_path_": image_path_
        }
        self._length = len(self.labels["image_path_"])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        img = self.rescaler(image=img)["image"]

        if self.augment:
            img = self.augmentation(image=img)["image"]

        example["image"] = (img / 127.5 - 1.0).astype(np.float32)
        return example

class CataractTrain(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CataractValidation(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)