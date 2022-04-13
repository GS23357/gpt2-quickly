from train import load_tokenizer
from multiprocessing import Process, Manager, Queue
from tqdm import tqdm
from typing import List
import pickle
import configs
import numpy as np
import click
import re


def encode_processer(processer_num, data):
    
    split_token_re=r"\[CLS\]"
    datas = re.split(split_token_re, data)
    
    max_length = configs.model.max_length
    tokenizer = load_tokenizer()
    input_ids = []
    size_right = 256
    real_size = max_length - size_right
    
    for i in tqdm(range(len(datas))):
        text = "".join(datas[i])
        if not text:
            continue
        
        text_encoded = tokenizer(text, return_attention_mask=False,
                                    return_token_type_ids=False, add_special_tokens=False)['input_ids']

        batch_size = len(text_encoded) // real_size
    
        for i in range(batch_size-1):
            current_encoded = text_encoded[real_size*i: real_size*(i+1)+size_right]
        
            assert len(current_encoded) == max_length
        
            input_ids.append(current_encoded)
            
        else:
            current_encoded = text_encoded[real_size*(i+1):]
            fill_last = np.zeros([max_length-len(current_encoded)], dtype=np.int).tolist()
            current_encoded += fill_last
        
            assert len(current_encoded) == max_length
        
            input_ids.append(current_encoded)
    

    print(f"Number of samples: {len(input_ids)}")

    input_ids = np.array(input_ids)
    input_ids = input_ids[1:]
    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    with open(configs.data.pickle.replace('.pickle', f'_{0}.pickle'), 'wb') as f:
        pickle.dump((ids, labels), f)


def preprocess():
    block_size = configs.model.max_length

    print(f'reading {configs.data.raw}')
    with open(configs.data.raw, 'r') as f:
        data = f.read()
        print(f"total words: {len(data)}")

    encode_processer(0, data, )


if __name__ == '__main__':
    preprocess()
