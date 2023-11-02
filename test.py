import torch
import torchvision
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import models.model as models
import models.utils as utils

def mapping(data):
    dict_ = {}
    for name, text in data:
        name = name.replace('/', '_')
        dict_[name] = text
    return dict_

def test_model(path, model):

	list_file = sorted(os.listdir(path))
	res_best_path = []
	res_beam_search = []

	tf = torchvision.transforms.Compose([
          torchvision.transforms.Resize((64, 768)),
		  torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225))
		])

	with torch.no_grad():
		for x in tqdm(list_file, ascii=True):
			img_path = './OCR/public_test/images/'
			x = x.replace('_', '/')

			img = torchvision.io.read_image(f'{img_path}{x}').float()
			img = tf(img)
			img = torch.unsqueeze(img, 0)
			out = model(img)
			out = out.permute(1, 0, 2)
			out = out[0].softmax(1).numpy()
				
			preds = utils.best_path_decoder(out, idx2char)
			preds_beam = utils.beam_search_decoder(out, 3, idx2char)
			
			res_beam_search.append([x.replace('_', '/'), preds_beam])
			res_best_path.append([x.replace('_', '/'), preds])

		return pd.DataFrame(data=res_beam_search, columns=['id', 'answer']), pd.DataFrame(data=res_best_path, columns=['id', 'answer'])



data = utils.get_img_name_and_labels('./OCR/training_data/annotations')
all_chars = utils.get_all_char(data[:,-1])
    
char2idx = utils.char2idx(all_chars)
idx2char = utils.idx2char(all_chars)

model = models.CRNN(imgH=64,nc=3, nh=256, nclass=len(char2idx))
model_name = 'crnn_3_channels'

model.load_state_dict(torch.load(f'./save_models/{model_name}.pth', map_location=torch.device('cpu')))
model.eval()

df1, df2 = test_model('./OCR/public_test/binary', model)

df1.to_csv(f'./submission/{model_name}_beam_submission.csv', index=False)
df2.to_csv(f'./submission/{model_name}_best_submission.csv', index=False)