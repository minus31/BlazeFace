import cv2
import pickle
import glob 
import os 
import numpy as np 


IM_EXTENSIONS = ['png', 'jpg', 'bmp']


def read_img(img_path, img_shape=(128,128)):
    """
    load image file and divide by 255.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_shape)
    img /= 255.

    return img


def dataloader(dataset_dir, label_path,  batch_size=32, img_shape=(128, 128)):

    """
    data loader

    return image, [class_label, class_and_location_label]
    """
    
    img_files = glob.glob(dataset_dir)
    img_files = [f for f in img_files if f[-3:] in IM_EXTENSIONS]

    with open(label_path, "rb") as f:
        labels = pickle.load(f)
    
    numofData = len(img_files)# endwiths(png,jpg ...)
    data_idx = np.arange(numofData)
    
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, replace=False)
        
        batch_img = []
        batch_label = []
        batch_label_cls = []
        
        for i in batch_idx:
            
            img = read_img(img_files[i], img_shape=img_shape)
            label = labels[i]
            
            batch_img.append(img)
            batch_label.append(label)
            batch_label_cls.append(label[0:1])
            
        yield np.array(batch_img, dtype=np.float32), 
        [np.array(batch_label_cls, dtype=np.float32), np.array(batch_label, dtype=np.float32)]


if __name__ == "__main__":
    pass

