# import modules
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
import warnings
import glob
import time
import os
import pickle

# image data preprocessing
class MyTransformer():
    """
    This class is a preprocessing for image data.
    First, this class load image data of ndarray, and resize this data to the size you select.
    Next, when attribute "phase" is "train", input image are transformed by the method "augmentation".

    Attributes
    ----------
    img_size (int):
        the size of image after preprocessing.
    ch (int):
        the number of third dimension of data.
        ex) time-serise data; "ch" in (image_size, image_size, ch) is time of snapshot.
            color or grayscale; ch=1 correspond to grayscale image, snd ch=3 is RGB color image.
    phase (str, choice=["train", "val", "test"]): 
        specify the phase of learning. When "train" id selected, image is transformed by the method "augmentation".
        When "val" is selected, there is no augmentation in preprocessing.
    """

    def __init__(self, img_size, ch=1, phase='train'):
        self.img_size = img_size
        self.ch = ch
        self.phase = phase

        if not phase in ['train', 'val', 'test']:
            raise ValueError('phase={} is unexpected.'.format(phase))
        
    
    def augmentation(self, image, rate=0.5, seed=None):
        '''
        This method do data augmentation such as rotation and flip for image。
        Parameters
        ----------
        image (float):
            input image. image.shape=(self.image_size, self.image_size, self.ch)
        rate (float): 
            probability that input image is transformed
            by each horizontal-flip, vertical-flip, 90 degrees rotation clockwise, and 90 degrees rotation counterclockwise
        seed (int):
            seed of random number. when you fix this, you can train machine in the same augmentation porcess.
        '''
        # fix seed of random number. As default, seed = None.
        np.random.seed(seed)
        
        # rotation of image
        if np.random.rand() < rate:
            if np.random.rand() > 0.5:
                # counterclockwise
                kk = -1
            else:
                # clockwise
                kk = 1
            image = np.rot90(image, k=kk)
        
        #  horizontal flip
        if np.random.rand() < rate:
            image = image[:,::-1,:]
        
        # vertical flip
        if np.random.rand() < rate:
            image = image[::-1,:,:]
            
        return image
        
                
    def __call__(self, file_path, rate=0.5, seed=None, debugmode=False):
        '''
        Parameters
        ----------
        file_path (str):
            file path to image
        rate, seed : 
            parameter for the 'augmentation' method
        debugmode (bool):
            when this param is True, warning is called in the case that shape of data is diffrent from (self.img_size, self.img_size, self.ch)
        '''
        
        ## load image data
        img = np.load(file_path)
        
        ## you can use only 2-dimensional or 3-dimensional data 
        if len(img.shape) < 2 or len(img.shape) > 3:
            raise ValueError('the size of input image is strange, shape of input: %s' % (img.shape,))
            
        ## if the dimension of the input image is two, the third dimension is added.
        if len(img.shape) == 2:
            img.resize(img.shape[0], img.shape[1], 1)
            
        ## resize image
        if not img.shape == (self.img_size, self.img_size, self.ch):
            if debugmode == True:
                print(file_path)
                warnings.warn("original size of input image is differnt from 'img_size' you selected. please check it.")
            if self.ch == 1:
                img = Image.fromarray(img.transpose(2,0,1)[0])
                img = img.resize((self.img_size, self.img_size))
            elif self.ch == 3:
                img = Image.fromarray(img)
                img = img.resize((self.img_size, self.img_size, self.ch))

            img = np.array(img)
            img.resize(self.img_size, self.img_size, self.ch)
            
        ### img.shape = (self.img_size, self.img_size, self.ch)
        if self.phase == 'train':
            # data augmentaion
            img = self.augmentation(img, rate=rate, seed=seed)
        
        # cast to float
        img = img.astype(np.float32)

        return img


# make the list of the set of filepath and label
def filenames_and_labels(data, phase='train'):
    """
    this function make the list of the set of filepath and label

    Parameters
    ----------
    data (file): csv text file and three columns.
            colum1 (int): label of category
            colum2 (str): name of category (not used in this function)
            colum3 (str): name of directory of data included. 
                you need directroies "colum3/train", "colum3/val" and "colum3/test"
    phase (str): "train", "val", or "test"
    """
    class_label, class_name, file_dir = np.loadtxt(data, unpack=True, dtype='object')
    
    # list of filepath of all data
    all_images = []
    # list of label of all data
    all_labels = []
    for i in range(len(class_label)):
        image_one_class = glob.glob(file_dir[i]+'/'+phase+'/*')

        if len(image_one_class) == 0:
            raise ValueError(file_dir[i]+'/'+phase, ' is empty.')

        all_images = all_images + image_one_class
        all_labels = all_labels + (np.zeros(len(image_one_class)) + int(class_label[i])).astype(np.int64).tolist()
        
    return all_images, all_labels


# construct data loader
class MyDataset(torch.utils.data.Dataset):
    """
    this class make dataset for pytorch. 
    input image is transformed by the way which is defined in 'transform' instance, and converted from ndarray to pytorch tensor.

    Attributes
    ----------
    file_names (str):
        the list of filenames of data
    labels (int):
        the list of labels of data
    transform (instance):
        the way to transform data. In this instance, you must define __call__ method where how to transform data are defined.
    """

    def __init__(self, file_names, labels, transform=None):
        self.file_names = file_names
        self.labels = labels
        self.transform = transform

    # when you use len(self), this method are called.
    def __len__(self):
        return len(self.file_names)

    # when you use self[idx], this method is called
    def __getitem__(self, idx):
        if self.transform == None:
            image = torch.tensor(np.load(self.file_names[idx]), dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.int64)

        # data preprocessing
        else:
            image = torch.tensor(self.transform(self.file_names[idx]), dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.int64)
            
        # shape of image input to CNN should be (batch_size, channel, height, width)
        # ref, https://tzmi.hatenablog.com/entry/2020/02/16/170928
        return image.permute(2,0,1), label


# function for training for one epoch
def train_epoch(model, optimizer, criterion, dataloader, device):
    '''
    parameters
    ----------
    model: instance of model
    optimizer: optimization method.
    criterion: loss function
    '''

    train_loss = 0.
    num_correct = 0.
    loss_count = 0.
    
    # trining mode
    model.train()
    
    for i, (images, labels) in enumerate(dataloader):
        # load images and labels and move these data on device
        images, labels = images.to(device), labels.to(device)

        # initialize gradient        
        optimizer.zero_grad()
        
        # get outputs from CNN
        outputs = model(images)

        # count data predicted as correct category
        for j in range(len(labels.cpu().numpy())):
            if np.argmax(outputs.detach().cpu().numpy()[j]) == labels.cpu().numpy()[j]:
                num_correct += 1.
        
        
        # calc loss
        loss = criterion(outputs, labels)
        # back propagation
        loss.backward()
        # optimization
        optimizer.step()
        
        train_loss += loss.item()
        loss_count += 1.
        
    # average loss for one epoch
    train_loss = train_loss / loss_count
    # calc accuracy
    acc = num_correct / len(dataloader.dataset)

    return train_loss, acc


# function for validation
def validation(model, criterion, dataloader, device):    
    # evaluation mode
    model.eval()
    
    test_loss=0
    num_correct = 0.
    loss_count = 0.

    # gradient is not calculated in this block
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            for j in range(len(labels.cpu().numpy())):
                if np.argmax(outputs.detach().cpu().numpy()[j]) == labels.cpu().numpy()[j]:
                    num_correct += 1.

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            loss_count += 1.
        test_loss = test_loss / loss_count
        acc = num_correct / len(dataloader.dataset)

    return test_loss, acc


# perform training
def run_training(num_epochs, model, train_loader, val_loader, optimizer, criterion, device, loss_list=[], output_dir='./results'):
    '''
    parameters
    ----------
    num_epochs: the number of training
    '''
    
    if len(loss_list) == 0:
      train_loss_list = []
      test_loss_list = []
      start_epoch = 0
      min_val = 1e10

    # you want to start where you stopped training
    else:
      train_loss_list, test_loss_list = loss_list
      min_val = np.min(test_loss_list)
      train_loss_list = train_loss_list.tolist()
      test_loss_list = test_loss_list.tolist()
      start_epoch = len(train_loss_list)

    for epoch in range(num_epochs):

        # when the file named as 'stop' exists, this program end training.
        if os.path.isfile(output_dir+'/stop'):
            print('get the call of stop. training is finished and save model.')
            break

        start = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = validation(model, criterion, val_loader, device)

        print(f'Epoch [{epoch+1+start_epoch}], time : {time.time()-start:.1f} s, train_loss : {train_loss:.4f}, train_acc : {train_acc:.3f}, val_loss : {test_loss:.4f}, val_acc : {test_acc:.3f}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if test_loss < min_val:
            min_val = test_loss
            #min_valloss_epoch = epoch+1+start_epoch
            with open(output_dir + '/min_val_model.p', 'wb') as f:
                pickle.dump(model, f)

    print('min_val_model is saved at epoch {}.'.format(np.argmin(np.array(test_loss_list))))

    return train_loss_list, test_loss_list


# function for test
def run_test(model, filename_list, output_file, dataloader, device):
    '''
    parameters
    ----------
    filename_list: list of filename
    '''
    model.eval()
    
    num_correct = 0.

    with torch.no_grad():
        with open(output_file, 'w') as f:
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                true_label = labels.cpu().numpy()
                predicted_label = np.argmax(outputs.detach().cpu().numpy())

                if true_label == predicted_label:
                    num_correct += 1.
                    t_or_f = 1                    
                else:
                    t_or_f = 0
                    
                f.write(filename_list[i] + '\n')
                f.write('true_label : {}\n'.format(true_label[0]))
                f.write('predicted_label: {}\n'.format(predicted_label))
                if t_or_f == 1:
                    f.write('true or false: true\n')
                if t_or_f == 0:
                    f.write('true or false: false\n')                    
                f.write('output from CNN (w/o softmax):\n')
                f.write(str(outputs.detach().cpu().numpy()[0]) + '\n')
                f.write('output from CNN (w/ softmax):\n')
                f.write(str(np.exp(outputs.detach().cpu().numpy()[0]) / np.sum(np.exp(outputs.detach().cpu().numpy()[0]))) + '\n\n')
                
        acc = num_correct / len(dataloader.dataset)

        print('test accuracy:', acc)
