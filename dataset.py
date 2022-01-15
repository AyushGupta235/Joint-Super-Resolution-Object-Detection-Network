import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

import config
import model

class ImageFolder(Dataset):
    def __init__(self, root_dir):
        super(ImageFolder, self).__init__()
        
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.ndarray(Image.open(os.path.join(dir, img)))

        return image