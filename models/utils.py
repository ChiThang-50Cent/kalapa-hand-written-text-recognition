import os
import numpy as np
import torch
import random
import cv2 as cv

text_path = 'annotations'
img_path = 'images'
binary = 'binary'
path = '../OCR/training_data/'

def split_to(text: str):
    text = text.replace('\n', '')
    text = text.split('\t')
    return text

def get_img_name_and_labels(text_path):
    data = []
    annotation_files = sorted(os.listdir(text_path))

    for name in annotation_files:
        if '.txt' in name:
            files = open(f'{text_path}/{name}', encoding='utf-8')
            lines = files.readlines()
            data.extend(list(map(split_to, lines)))

    return np.array(data)

def get_all_char(data: np.ndarray):
    all_labels = "".join(data)

    all_labels_1 = all_labels.upper()
    all_labels_2 = all_labels.lower()

    res = sorted(list(set(list(all_labels_1 + all_labels_2))))

    return res

def char2idx(list_char: list):
    c_2_i = {c : i + 1 for i, c in enumerate(list_char)}
    c_2_i['-'] = 0
    return c_2_i

def idx2char(list_char: list):
    i_2_c = {i + 1 : c for i, c in enumerate(list_char)}
    i_2_c[0] = '-'
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

def convert_img(data, img_path, save_path):
    for i, (dir, title) in enumerate(data):

        img = cv.imread(f'{img_path}/{dir}')

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

        name = dir.replace('/', '_')
        check = cv.imwrite(f'{save_path}/{name}.jpg', label)

        print(i, name) if check else print(i, name, 'Error')

if __name__ == '__main__':

    text_batch = ['Một hai ba', 'Mot hai']
    data = get_all_char(text_batch)
    c2i = char2idx(data)

    print(encode_target_batch(text_batch, c2i))
    