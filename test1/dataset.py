from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import scipy.io as sio 
import time
class IMAGE_Dataset(Dataset):
    
    def __init__(self, root_dir,matfn, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.data = sio.loadmat(matfn)['annotations']
        self.transform = transform
        for i in range(len(self.data[0])):
                self.x.append('../../'+str(self.root_dir)+"/"+str(self.data[0][i][4][0][0])+"/"+str(self.data[0][i][5][0]))
                self.y.append(int(self.data[0][i][4][0][0]))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open('../../'+str(self.root_dir)+"/"+str(self.data[0][index][4][0][0])+"/"+str(self.data[0][index][5][0])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.data[0][index][4][0][0])
