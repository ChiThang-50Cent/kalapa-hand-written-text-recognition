import torch
import torchvision

import models.model as models
import models.utils as utils

def mapping(data):
    dict_ = {}
    for name, text in data:
        name = name.replace('/', '_')
        dict_[name] = text
    return dict_

data = utils.get_img_name_and_labels('./OCR/training_data/annotations')
all_chars = utils.get_all_char(data[:,-1])
    
char2idx = utils.char2idx(all_chars)
idx2char = utils.idx2char(all_chars)

model = models.CRNN(imgH=64,nc=1, nh=256, nclass=len(char2idx))
model.load_state_dict(torch.load('./crnn.pth', map_location=torch.device('cpu')))

img = torchvision.io.read_image('./OCR/training_data/binary/300_26.jpg')

tf = torchvision.transforms.Resize((64, 768))
img = tf(img)

img = torch.unsqueeze(img, 0)

out = model(img)
out = out.permute(1, 0, 2)

name_text = mapping(data)

print(name_text['300_26.jpg'])
print(utils.decode_sentence(out[0].softmax(1).argmax(1).numpy(), idx2char))