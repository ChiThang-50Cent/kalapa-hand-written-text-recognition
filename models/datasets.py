import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import models.utils as utils

text_path = 'annotations'
img_path = 'images'
path = '../OCR/training_data/'

class ImageFolders(Dataset):
  def __init__(self, path=path, text=text_path, image=img_path, transform=None):
    super().__init__()
    img_path_and_label = utils.get_img_name_and_labels(
      f'{path}{text}')
    self.names = img_path_and_label[:,0]
    self.labels = img_path_and_label[:,-1]

    self.transform = transform
    self.path = path
    self.image = image

  def __len__(self):
    return len(self.names)

  def __getitem__(self, index):
    img = torchvision.io.read_image(
      f'{self.path}{self.image}/{self.names[index]}'
    )
    
    label = self.labels[index]

    if self.transform:
      img = self.transform(img)

    return img, label

compose = transforms.Compose([
  utils.to_binary_transform(),
  transforms.Resize((64, 768)),
  transforms.RandomRotation(3),
  transforms.ElasticTransform(7.0),
  transforms.RandomErasing(),
])

if __name__ == '__main__':
  dataset = ImageFolders(transform=compose)
  print(torch.min(dataset[0][0]), torch.max(dataset[0][0]))
