import os
import numpy as np
import torch
import random
import cv2 as cv

text_path = 'annotations'
img_path = 'images'
path = '../OCR/training_data/'

def split_to(text: str):
    text = text.replace('\n', '')
    text = text.split('\t')
    return text

def get_img_name_and_labels(text_path):
    data = []
    annotation_files = os.listdir(text_path)

    for name in annotation_files:
        if '.txt' in name:
            files = open(f'{text_path}/{name}', encoding='utf-8')
            lines = files.readlines()
            data.extend(list(map(split_to, lines)))

    return np.array(data)

def get_all_char(data: np.ndarray):
    all_labels = " ".join(data).split(' ')

    all_labels_1 = "".join(all_labels).upper()
    all_labels_2 = "".join(all_labels).lower()

    res = sorted(list(set(list(all_labels_1 + all_labels_2))))

    return res

def char2idx(list_char: list):
    c_2_i = {c : i for i, c in enumerate(list_char)}
    c_2_i[' '] = 197
    c_2_i['-'] = 198
    return c_2_i

def idx2char(list_char: list):
    i_2_c = {i : c for i, c in enumerate(list_char)}
    i_2_c[197] = ' '
    i_2_c[198] = '-'
    return i_2_c

def encode_sentence(sentences: str, char2idx: dict):
    encoded = [char2idx[s] for s in sentences]
    return encoded

def decode_sentence(sentences: list, idx2char: dict):
    decoded = [idx2char[s] for s in sentences]
    return "".join(decoded)

def encode_target_batch(target_batch, char2idx):
    
    text_batch_targets_lens = [len(text) for text in target_batch]
    text_batch_targets_lens = torch.tensor(text_batch_targets_lens)
    
    text_batch_concat = "".join(target_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.tensor(text_batch_targets)
    return text_batch_targets, text_batch_targets_lens

def compute_loss(criterion, target_batch, text_batch_logits, device, char2idx):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = text_batch_logits
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.int32).to(device) # [batch_size] 
     
    text_batch_targets, text_batch_targets_lens = encode_target_batch(target_batch, char2idx)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss

def all_seed(var_seed):
    np.random.seed(var_seed)
    random.seed(var_seed)
    torch.manual_seed(var_seed)

def convert_img(data, path, save_path):

  for i, (dir, title) in enumerate(data):

    print(i, dir)
    img = cv.imread(f'{path}/{dir}')

    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + 
                cv.TERM_CRITERIA_MAX_ITER, 
                10, 1.0)
    
    _, label, _1=cv.kmeans(Z, 2, None, criteria, 
                            10, cv.KMEANS_RANDOM_CENTERS)

    if np.sum(label) / len(label) < 0.5:
      label = 1 - label
    
    label = label.reshape(img.shape[0], img.shape[1], 1)

    cv.imwrite(f'{save_path}/{title}.jpg', label)
  
if __name__ == '__main__':

    data = get_img_name_and_labels(f'{path}{text_path}')
    list_char = get_all_char(data[:,-1])

    char_2_idx = char2idx(list_char)
    idx_2_char = idx2char(list_char)

    out = encode_target_batch(['Ấp Cầu Ngang Long Hữu Đông Cần Đước Long An Phan Hoai Nhan Falsc Jpq'], char_2_idx)
    print(out)