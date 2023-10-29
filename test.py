import torch
import torchvision

import models.model as models
import models.utils as utils

all_chars = utils.get_all_char(
        utils.get_img_name_and_labels('./OCR/training_data/annotations')[:,-1])
    
char2idx = utils.char2idx(all_chars)
idx2char = utils.idx2char(all_chars)

model = models.Model((64, 768), 'vgg16', 1, 256, len(char2idx))
model.load_state_dict(torch.load('./vgg16_256.pth', map_location=torch.device('cpu')))

img = torchvision.io.read_image('./OCR/training_data/binary/300_26.jpg')

tf = torchvision.transforms.Resize((64, 768))
img = tf(img)

img = torch.unsqueeze(img, 0)

out = model(img)
out = out.permute(1, 0, 2)

print(utils.decode_sentence(out[0].softmax(1).argmax(1).numpy(), idx2char))