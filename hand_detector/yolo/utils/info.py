import os


def data_info(type='train'):
    dataset_directory = 'custom_dataset/'
    folder_names = ['train']

    if type is 'train':
        folder_names = ['train']
        nTrain = 0
        for folder in folder_names:
            nTrain = nTrain + len(os.listdir(dataset_directory + folder + '/'))
        return nTrain

    elif type is 'valid':
        folder_names = ['valid']
        nValid = 0
        for folder in folder_names:
            nValid = nValid + len(os.listdir(dataset_directory + folder + '/'))
        return nValid

    elif type is 'test':
        folder_names = ['test']
        nTest = 0
        for folder in folder_names:
            nTest = nTest + len(os.listdir(dataset_directory + folder + '/'))
        return nTest
