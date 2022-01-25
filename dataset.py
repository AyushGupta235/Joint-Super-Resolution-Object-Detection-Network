import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision

import config
import model

class ImageFolder(Dataset):
    def __init__(self, root_dir):
        super(ImageFolder, self).__init__()
        
        self.img_data = []
        self.hr_data = []
        self.label_data = []
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir, "images")
        self.label_path = os.path.join(self.root_dir, "labels")

        self.img = os.path.join(self.img_path, config.MODE)
        self.labels = os.path.join(self.label_path, config.MODE)
        self.hr = os.path.join(self.img_path, "hr")

        index = 0
        for f in sorted(os.listdir(self.img)):
            image = os.path.join(self.img, f)
            self.img_data += list(zip([image], [index] * len(image)))
            index += 1

        index = 0
        for f in sorted(os.listdir(self.labels)):
            label = os.path.join(self.labels, f)
            self.label_data += list(zip([label], [index] * len(label)))
            index += 1

        index = 0
        for f in sorted(os.listdir(self.hr)):
            image = os.path.join(self.hr, f)
            self.hr_data += list(zip([image], [index] * len(image)))
            index += 1


    def __len__(self):
        return len(self.img_data)


    def __getitem__(self, index):
        trans = torchvision.transforms.ToTensor()

        img, _ = self.img_data[index]
        image = Image.open(img)
        low_res_image = trans(image)

        hr, _ = self.hr_data[index]
        image = Image.open(hr)
        high_res_image = trans(image)
        
        label_path, _ = self.label_data[index]
        label = np.loadtxt(fname = label_path, delimiter = " ", ndmin = 2)

        return low_res_image, high_res_image, label