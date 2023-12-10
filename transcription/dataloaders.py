import numpy as np
from glob import glob
from os.path import join as ospj
from torch.utils.data import Dataset

class TranscriptionDataset(Dataset):
    CHUNKS_PATH = '/ssd/thiago.poppe/IDMT-SMT-Drums/chunks/'
    
    def __init__(self, is_train: bool):
        split = 'train' if is_train else 'validation'
        self.filenames = glob(ospj(self.CHUNKS_PATH, split, '*.npz'))
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        data = np.load(self.filenames[idx])
        return data['spec'].T, data['annotation'].T