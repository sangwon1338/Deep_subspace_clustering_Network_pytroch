from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio

class COIL20(Dataset):
    def __init__(self, transform, path):
        # Transforms
        self.transfrom = transform

        self.data = sio.loadmat(path)

        self.image_arr = self.data['fea']
        self.label_arr = self.data['gnd']
        self.data_len = self.data['fea'].shape[0]
       
        

    def __getitem__(self, idx):
        img=self.image_arr[idx]
        img=np.reshape(img,(32,32,1))
        img=img.astype(np.float32)
        img=self.transfrom(img)

        label=self.label_arr[idx][0]
        

        return (img, label)

    def __len__(self):
        return self.data_len