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
  def __init__(self, root, img_path, text_path, binary = 1, transform=None):
    super().__init__()

    self.img_path = f'{root}{img_path}'
    self.binary = bool(binary)
    self.img_labels = utils.get_img_name_and_labels(f'{root}{text_path}')
    self.transform = transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, index):
    img = str(self.img_labels[index][0]) 
    if self.binary:
      img = img.replace('/', '_')

    img = torchvision.io.read_image(
      f'{self.img_path}/{img}'
    )
    
    img = img.float()

    label = self.img_labels[index][1]

    if self.transform:
      img = self.transform(img)

    return img, label

def compose(n_channels, binary=False):
  transform = [
    transforms.Resize((64, 768)),
    transforms.RandomRotation(3),
    transforms.ElasticTransform(7.0),
    transforms.RandomErasing(),
  ]
  if n_channels != 1:
    transform.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225)))
  elif not bool(binary):
    transform.insert(1, transforms.Grayscale())
    transform.append(transforms.Normalize(mean=(0.5,),
                                           std= (0.5,)))


  return transforms.Compose(transform)

if __name__ == '__main__':
  dataset = ImageFolders(root=path,
                         img_path=img_path,
                         text_path=text_path,
                         transform=compose(n_channels=3))
  print(torch.min(dataset[0][0]), torch.max(dataset[0][0]))
