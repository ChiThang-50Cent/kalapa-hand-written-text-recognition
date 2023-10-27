import os
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import models.utils as utils

text_path = 'annotations'
img_path = 'images'
path = '../OCR/training_data/'

class ImageFolders(Dataset):
  def __init__(self, path=path, img_path=None, transform=None):
    super().__init__()

    self.path = f'{path}{img_path}'
    self.list_img = os.listdir(self.path)
    self.transform = transform

  def __len__(self):
    return len(self.list_img)

  def __getitem__(self, index):
    img = torchvision.io.read_image(
      f'{self.path}/{self.list_img[index]}'
    )
    
    label = self.list_img[index][:-4]

    if self.transform:
      img = self.transform(img)

    return img, label

compose = transforms.Compose([
  transforms.Resize((64, 768)),
  transforms.RandomRotation(3),
  transforms.ElasticTransform(7.0),
  transforms.RandomErasing(),
])

if __name__ == '__main__':
  dataset = ImageFolders(transform=compose)
  print(torch.min(dataset[0][0]), torch.max(dataset[0][0]))
