import torch
import torchvision

import model as models
import utils as utils

class Predictor():
    def __init__(self) -> None:
        self.tf = torchvision.transforms.Compose([
          torchvision.transforms.Resize((64, 768)),
		  torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5))
		])

        data = utils.get_img_name_and_labels('../OCR/training_data/annotations/')
        all_chars = utils.get_all_char(data[:,-1])
            
        char2idx = utils.char2idx(all_chars)

        self.idx2char = utils.idx2char(all_chars)

        model = models.CRNN(imgH=64,nc=3, nh=256, nclass=len(char2idx))
        model_name = 'crnn_3_channels'

        model.load_state_dict(torch.load(f'../save_models/{model_name}.pth', map_location=torch.device('cpu')))

        self.model = model

    def encode(self, img):
        with torch.no_grad():
            img = torchvision.io.read_image(img).float()
            img = self.tf(img)
            img = torch.unsqueeze(img, 0)
            out = self.model(img)
            out = out.permute(1, 0, 2)
            out = out[0].softmax(1).numpy()
            return out

    def decode(self, encoded):
        preds = utils.best_path_decoder(encoded, self.idx2char)
        preds_beam = utils.beam_search_decoder(encoded, 3, self.idx2char)

        return preds, preds_beam