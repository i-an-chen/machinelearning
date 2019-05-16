import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
CUDA_DEVICES = 0
DATASET_ROOT = './cars_test_crop'
DATASET_mat = 'cars_test_annos_withlabels.mat'
PATH_TO_WEIGHTS = './1model-10-best_train_acc.pth'


def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT),DATASET_mat, data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
    classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
   
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            # batch size
            for i in range(labels.size(0)):
                label =labels[i]-1
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy on the ALL test images: %d %%'
          % (100 * total_correct / total))

    for i, c in enumerate(classes):
        print('Accuracy of %5s : %2d %%' % (
        c, 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    test()
    PATH_TO_WEIGHTS= './1model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './1model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './1model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './1model-50-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './2model-10-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './2model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './2model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './2model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './2model-50-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './3model-10-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './3model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './3model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './3model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './3model-50-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './4model-10-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './4model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './4model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './4model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './4model-50-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './5model-10-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './5model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './5model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './5model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './5model-50-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './6model-10-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './6model-20-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './6model-30-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './6model-40-best_train_acc.pth'
    test()
    PATH_TO_WEIGHTS= './6model-50-best_train_acc.pth'
    test()
