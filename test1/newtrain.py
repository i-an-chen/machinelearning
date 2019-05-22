import torch
import torch.nn as nn
import torchvision.models as models
from resnet import myCNN
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy


##REPRODUCIBILITY
torch.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path

CUDA_DEVICES = 0
DATASET_ROOT = 'cars_train_crop'
DATASET_mat = 'cars_train_annos.mat'


def train():
	resnet50 = models.resnet50(pretrained=True)
	cnn = myCNN()	
	
	pretrained_dict = resnet50.state_dict()
	model_dict = cnn.state_dict()
	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	cnn.load_state_dict(model_dict)
	
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	train_set = IMAGE_Dataset(Path(DATASET_ROOT),DATASET_mat, data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=50, shuffle=True, num_workers=1)
	cnn = cnn.cuda(CUDA_DEVICES)
	cnn.train()
	
	classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
	#best_model_params = copy.deepcopy(cnn.state_dict())
	best_acc = 0.0
	num_epochs = 50
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=cnn.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):

			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = cnn(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')


		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(cnn.state_dict())
		if (epoch+1)%10 == 0 :
			cnn.load_state_dict(best_model_params)
			torch.save(cnn, f'train_acc.pth')


if __name__ == '__main__':
	train()
