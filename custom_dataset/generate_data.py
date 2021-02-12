import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import random
import cv2

# Function to extract list of frames
def FrameCapture(path_to_video, video_name, train_frame_indices, valid_frame_indices, video_format = '.mp4'):

    # Path to video file
    print('Process : ', path_to_video+'/'+video_name + video_format)
    vidObj = cv2.VideoCapture(path_to_video+'/'+video_name + video_format)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    max_nb_frame = max(max(train_frame_indices), max(valid_frame_indices))
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        if count in train_frame_indices:
            print('to train')
            cv2.imwrite( 'train/v_{}_frame_{}.jpg'.format(video_name, count), image)
            
        elif count in valid_frame_indices:
            print('to valid')
            cv2.imwrite( 'valid/v_{}_frame_{}.jpg'.format(video_name, count), image)
            
        if count > max_nb_frame:
            break
        count += 1


def xml_to_csv(video_name, total_train_labels, total_valid_labels, valid_prob):
    train_frame_indices, valid_frame_indices = [],[]
    if video_name == '15':
        for xml_file in glob.glob('annotations' + '/' + video_name + '/Annotations/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for member in root.findall('object'):
                frame_index = int(root.find('filename').text[6:])
                value = ('v_{}_frame_{}'.format(video_name, frame_index),
                         0,
                         0,
                         0,
                         0
                         )
                
                if random.random() < valid_prob:
                    total_valid_labels.append(value)
                    valid_frame_indices.append(frame_index)
                else:
                    total_train_labels.append(value)
                    train_frame_indices.append(frame_index)
    else: 
        for xml_file in glob.glob('annotations' + '/' + video_name + '/Annotations/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for member in root.findall('object'):
                frame_index = int(root.find('filename').text[6:])
                value = ('v_{}_frame_{}'.format(video_name, frame_index),
                         float(member[2][0].text)/int(root.find('size')[0].text),
                         float(member[2][1].text)/int(root.find('size')[1].text),
                         float(member[2][2].text)/int(root.find('size')[0].text),
                         float(member[2][3].text)/int(root.find('size')[1].text)
                         )
                
                if random.random() < valid_prob:
                    total_valid_labels.append(value)
                    valid_frame_indices.append(frame_index)
                else:
                    total_train_labels.append(value)
                    train_frame_indices.append(frame_index)
    
    FrameCapture('videos', video_name, train_frame_indices, valid_frame_indices)   

    return total_train_labels, total_valid_labels


# Function to extract list of frames
def FrameCapture_t(path_to_video, video_name, test_frame_indices, video_format = '.mp4'):

    # Path to video file
    print('Process : ', path_to_video+'/'+video_name + video_format)
    vidObj = cv2.VideoCapture(path_to_video+'/'+video_name + video_format)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    max_nb_frame = max(test_frame_indices)
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        if count in test_frame_indices:
            print('to test')
            cv2.imwrite( 'test/v_{}_frame_{}.jpg'.format(video_name, count), image)
            
        if count > max_nb_frame:
            break
        count += 1


def xml_to_csv_t(video_name, total_test_labels):
    test_frame_indices = []
    if video_name == 'test_3':
        for xml_file in glob.glob('annotations' + '/' + video_name + '/Annotations/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for member in root.findall('object'):
                frame_index = int(root.find('filename').text[6:])
                value = ('v_{}_frame_{}'.format(video_name, frame_index),
                         0,
                         0,
                         0,
                         0
                         )
                

                total_test_labels.append(value)
                test_frame_indices.append(frame_index)

    else:      
        for xml_file in glob.glob('annotations' + '/' + video_name + '/Annotations/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for member in root.findall('object'):
                frame_index = int(root.find('filename').text[6:])
                value = ('v_{}_frame_{}'.format(video_name, frame_index),
                         float(member[2][0].text)/int(root.find('size')[0].text),
                         float(member[2][1].text)/int(root.find('size')[1].text),
                         float(member[2][2].text)/int(root.find('size')[0].text),
                         float(member[2][3].text)/int(root.find('size')[1].text)
                         )
                

                total_test_labels.append(value)
                test_frame_indices.append(frame_index)

    
    FrameCapture_t('videos', video_name, test_frame_indices)   

    return total_test_labels



if __name__ == '__main__':
    # TRAIN + VALIDATION DATASETs
    n_videos = 14 
    video_names = ['{}.mp4'.format(i) for i in range(1,n_videos+1)]
    train_folder = '/train'
    valid_folder = '/valid'
    total_train_labels, total_valid_labels = [],[]
    valid_prob = 0.1
    for n in range(1,n_videos + 1):
        total_train_labels, total_valid_labels = xml_to_csv('{}'.format(n), total_train_labels, total_valid_labels, valid_prob)
    
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax']
    train_df = pd.DataFrame(total_train_labels, columns=column_name)
    valid_df = pd.DataFrame(total_valid_labels, columns=column_name)

    train_df.to_csv('train_labels.csv', index=None)
    valid_df.to_csv('valid_labels.csv', index=None)

    # # TEST DATASET
    # n_videos = 3
    # video_names = ['test_{}'.format(i) for i in range(1,n_videos+1)]
    # test_folder = '/test'
    # total_test_labels = []
    # for video_name in video_names:
    #     total_test_labels = xml_to_csv_t(video_name, total_test_labels)
    
    # column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax']
    # test_df = pd.DataFrame(total_test_labels, columns=column_name)
    # test_df.to_csv('test_labels.csv', index=None)

