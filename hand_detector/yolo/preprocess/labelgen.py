import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hand_detector.yolo.utils.utils import visualize
from hand_detector.yolo.preprocess.yolo_flag import Flag

f = Flag()

grid = f.grid
grid_size = f.grid_size
target_size = f.target_size
threshold = f.threshold

df_train = pd.read_csv('custom_dataset/train_labels.csv') 
df_valid = pd.read_csv('custom_dataset/valid_labels.csv') 
df_test = pd.read_csv('custom_dataset/test_labels.csv') 
# def label_generator(directory, image_name, type=''):
#     folder_name = find_folder(image_name)
#     image = plt.imread(directory + folder_name + type + '/' + image_name)
#     image = cv2.resize(image, (target_size, target_size))

#     file = open(directory + 'label/' + folder_name + '.txt')
#     lines = file.readlines()
#     file.close()

#     label = []
#     for line in lines:
#         line = line.strip().split()
#         name = line[0].split('/')[3]
#         if image_name == name:
#             label = line[1:]
#             break

#     """ bbox: top-left and bottom-right coordinate of the bounding box """
#     label = label[0:4]
#     bbox = [float(element) * target_size for element in label]
#     bbox = np.array(bbox)
#     return image, bbox

def label_generator(directory, folder, image_name):
    image = plt.imread(directory + folder + '/' + image_name)
    #image = cv2.resize(image, (target_size, target_size))

    if folder == 'train':
        df = df_train
    elif folder == 'valid':
        df = df_valid
    elif folder == 'test':
        df = df_test
    else:
        exit(1)
    """ bbox: top-left and bottom-right coordinate of the bounding box """
    label = df[df.filename == image_name[:-4]].iloc[0][1:].tolist()
    bbox = [float(element) * target_size for element in label]
    bbox = np.array(bbox)
    return image, bbox


def bbox_to_grid(bbox):
    if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
        output = np.zeros(shape=(grid, grid, 5))
    else:
        output = np.zeros(shape=(grid, grid, 5))
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        i, j = int(np.floor(center[0] / grid_size)), int(np.floor(center[1] / grid_size))
        i = i if i < f.grid else f.grid - 1
        j = j if j < f.grid else f.grid - 1
        output[i, j, 0] = 1
        output[i, j, 1:] = bbox / f.target_size
    return output


if __name__ == '__main__':
    img_name = 'v_1_frame_000418.jpg'
    dir = 'custom_dataset/'
    img, box = label_generator(directory=dir, folder= 'valid', image_name=img_name)
    yolo_out = bbox_to_grid(box)
    visualize(img, yolo_out, title='', RGB2BGR=True)
