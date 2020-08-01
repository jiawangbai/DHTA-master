import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class Dataset(data.Dataset):
    def __init__(self, root, data_name="imagenet", reproduce=True):

        self.data_name = data_name
        self.type = type

        data_path = []
        labels = []

        data_info = open(os.path.join("data", data_name, "query.txt"))
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(os.path.join(root, line_split[0]))
            labels.append(np.array(line_split[1:]).astype(float).tolist())

        # if true, load attack sample indexes to reproduce our result.
        # else, choose randomly.
        if reproduce:
            index = np.loadtxt("reproduce/attack_query_index.txt").astype(int)
        else:
            index = np.random.choice(len(data_path), 100, replace=False)
        self.data_list = np.array(data_path)[index].tolist()
        self.labels = np.array(labels)[index].tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        label = torch.tensor(self.labels[index])

        return img, torch.tensor(label)

    def transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.type == "train":
            return transforms.Compose([
                ResizeImage(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])(img)
        else:
            start_center = (256 - 224 - 1) / 2
            return transforms.Compose([
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                normalize
            ])(img)