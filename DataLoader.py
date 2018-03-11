# -*- coding:utf-8 -*-

# change!
import os
# change! use glob to return a list, which contains the all images.
import glob
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
from PIL import Image

# change! delete bsd500() method.

def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])

def get_training_set(size, target_mode='seg', colordim=1):
    # change! use train_dir.
    train_dir = "/root/datasets/CNN/Training_Set"
    # assert
    assert train_dir, "Please put the datasets in the floder"
    return DatasetFromFolder(train_dir, target_mode, colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))

def get_test_set(size, target_mode='bon', colordim=1):
    # change the get_test_set, refer to get_training_set().
    # can be refer to training_dataset to struct the testing datasets.
    # the training_dataset and testing_dataset structure could be same, in this way, the class DatasetFromFolder can
    # be reuse!
    # change! use train_dir.
    test_dir = "/root/datasets/CNN/Testing1_Set"
    # assert
    assert test_dir, "Please put the datasets in the floder"
    return DatasetFromFolder(test_dir, target_mode, colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, colordim):
    if colordim == 1:
        img = Image.open(filepath).convert('L')
        # img.size return the (W, H).
    else:
        img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_mode, colordim, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.target_mode = target_mode
        self.colordim = colordim
        
        # change! self.image_filenames
        # image_dir, e.g., is "./datasets/training_dataset", so listdir(image_dir) is a list, which contain the all 
        # folder and files. e.g., listdir(image_dir) is ['2092.jpg', ...], it only return the folder and files name.

        # Here, load ground_truth folder. x, e.g. is 
        # './datasets/training_dataset/ground_truth/patient1/P01dicom/P01-0080.jpg', 
        # so use slpit to get the 'patient1/P01dicom/P01-0080.jpg'.
        # Utilize is_image_file() judge the files is a image or not.

        # TODO: There must be a efficient way to get the image name(e.g., 'patient1/P01-0080.jpg')
        # or, you can put all the ground truth images to a txt file, then read txt file.
        self.image_filenames = [os.path.join(x.split("/")[-3:][0], x.split("/")[-3:][1], x.split("/")[-3:][2]) 
            for x in glob.glob(os.path.join(self.image_dir, "ground_truth", "*/*/*.*")) if is_image_file(x)]
        # print(len(self.image_filenames))

    def __getitem__(self, index):
        # change! load_img method argument(filepath) is a image path, e.g., 
        # './datasets/training_dataset/ground_truth/patient1/P01-0080.jpg'.
        # change all the load_img()!
        input_batch = load_img(os.path.join(self.image_dir, "images", self.image_filenames[index]), self.colordim)
        if self.target_mode == 'seg':
            target = load_img(os.path.join(self.image_dir, "mask", self.image_filenames[index]), 1)
        else:
            # TODO: If there are other tasks, it could be load data in here.
            pass

        if self.input_transform:
            input_batch = self.input_transform(input_batch)
        if self.target_transform:
            target = self.target_transform(target)

        return input_batch, target
        # return tuple. in EQtrain.py, it can use batch[0] and batch[1] to get input_batch, target.

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    # test load data is right or not?
    get_training_set(size=256)
