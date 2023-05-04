# import modules
import numpy as np
import os
import pickle
import time

### parameters ###
# data
model_name = 'cnn_model'
input_file = './dataset.txt'
output_result = './results'
test_file = 'test.txt'

'''
- CNN model is saved in 'output_result' as 'model_name'.
- Input data need to be the directory written in 'input_file'
and training, validation, and test data are stocked in 'train', 'val' and 'test', respectively.
- when 'prediction' parameter below is True, the outputs from CNN are written in 'test_file' in 'output_result'.
'''

# machine
# 0-3
gpu = '0' # this parameter depends on your machine.

# input image
img_size = 128
batch_size = 8
ch = 1

# params pof machie learning
num_epochs = 30
lr = 0.00001
weight_decay = 5e-5 # L2 regularization

## when below parameter is filepath to saved model, 
## training continue from where training left off last time.
 load_model = None
#load_model = output_result+'/'+model_name+'.pkl'

# when you want to test CNN, please switch below parameter to True.
prediction = False
#load_model = output_result+'/min_val_model.p'

#####################################################
# select gpu
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import torch
torch.backends.cudnn.benchmark = True

# print version od pytorch
total_time_start = time.time()
print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

if model_name == 'min_val_model':
    raise ValueError('Sorry, please choose other model name.')

# import my modules
from utils import MyTransformer, MyDataset, filenames_and_labels
from utils import run_training, run_test

print('data loading follows {}.'.format(input_file))

# make dataloader for training and validation
images_train, labels_train = filenames_and_labels(input_file, phase='train')
images_val, labels_val = filenames_and_labels(input_file, phase='val')

transform_train = MyTransformer(img_size=int(img_size), ch=int(ch), phase='train')
dataset_train = MyDataset(file_names=images_train, labels=labels_train, transform=transform_train)

transform_val = MyTransformer(img_size=int(img_size), ch=int(ch), phase='val')
dataset_val = MyDataset(file_names=images_val, labels=labels_val, transform=transform_val)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(batch_size), shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

# constract CNN
from cnn_model import myCNN
from torchinfo import summary

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model
if load_model != None:
    if os.path.isfile(load_model) == True:
        print('laod model {} and restart traning.'.format(load_model))
        print('learning rate : {}'.format(lr))
        with open(load_model, 'rb') as f:
            model = pickle.load(f)
            model = model.to(device)
        loss_list = np.load(output_result + '/loss_list.npy')

    else:
        print('{} does not exist. traning start from the begining.'.format(load_model))
        print('learning rate : {}'.format(lr))
        model = myCNN().to(device)
        print('==== model architecture ===')
        summary(model, (int(batch_size), int(ch), int(img_size), int(img_size)), col_names=["output_size", "num_params"])
        loss_list = []

else:
    print('traning start from the begining.')
    print('learning rate : {}'.format(lr))
    model = myCNN().to(device)
    print('==== model architecture ===')
    summary(model, (int(batch_size), int(ch), int(img_size), int(img_size)), col_names=["output_size", "num_params"])
    loss_list = []

# loss function and optimizer
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

momentum = 0
# weight_decay = 0 # L2 regularization
dampening = 0
nesterov = False
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, 
#                            weight_decay=weight_decay, nesterov=nesterov)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

if prediction == False:
    train_loss_list, test_loss_list = run_training(num_epochs=num_epochs,
                                               model=model,
                                               train_loader=train_loader,
                                               val_loader=val_loader,
                                               optimizer=optimizer,
                                               criterion=criterion,
                                               device=device,
                                               loss_list=loss_list,
                                               output_dir=output_result)

    with open(output_result + '/' + model_name + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    np.save(output_result + '/loss_list.npy', np.array([train_loss_list, test_loss_list]))

    print('trained model is saved in {}'.format(output_result+'/'+model_name+'.pkl'))

elif prediction == True:
    print('training is skipped. run test.')
    images_test, labels_test = filenames_and_labels(input_file, phase='test')
    transform_test = MyTransformer(img_size=int(img_size), ch=int(ch), phase='test')
    dataset_test = MyDataset(file_names=images_test, labels=labels_test, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    run_test(model=model, filename_list=images_test, output_file=output_result+'/'+test_file, dataloader=test_loader, device=device)

else:
    raise ValueError('prediction should be True or False.')
    
print(f'total time: {time.time()-total_time_start:.1f}s \n')

