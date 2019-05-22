from PIL import Image
import scipy.io as sio 
import numpy as np
import os
train_mat = 'cars_train_annos.mat'
test_mat = 'cars_test_annos_withlabels.mat'

def IMAGE_CROP():
    train = sio.loadmat(train_mat)['annotations']
    test = sio.loadmat(test_mat)['annotations']
    for i in range(len(train[0])):
            train_image = Image.open("./cars_train/"+train[0][i][5][0])
            train_image=train_image.crop((train[0][i][0][0][0], train[0][i][1][0][0], train[0][i][2][0][0], train[0][i][3][0][0]))
            directory = os.path.dirname("./cars_train_crop/"+str(train[0][i][4][0][0])+"/")
            if not os.path.exists(directory):
                os.makedirs(directory)
            train_image.save("./cars_train_crop/"+str(train[0][i][4][0][0])+'/'+train[0][i][5][0])
    for i in range(len(test[0])):
            test_image = Image.open("./cars_test/"+test[0][i][5][0])
            test_image=test_image.crop((test[0][i][0][0][0], test[0][i][1][0][0], test[0][i][2][0][0], test[0][i][3][0][0]))
            directory = os.path.dirname("./cars_test_crop/"+str(test[0][i][4][0][0])+"/")
            if not os.path.exists(directory):
                os.makedirs(directory)
            test_image.save("./cars_test_crop/"+str(test[0][i][4][0][0])+'/'+test[0][i][5][0])

if __name__ == '__main__':
	IMAGE_CROP()
