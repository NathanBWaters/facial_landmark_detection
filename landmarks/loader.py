'''
Pytorch character loader dataset class
'''
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd

from landmarks.constants import (
    TRAINING_DATA,
    TEST_DATA,
    INPUT_DIMS,
)


def csv_to_dataframe(csv_file):
    '''
    Converts a csv file to a dataframe
    '''
    df = pd.read_csv(csv_file)

    # we have data we need to fill in
    df.fillna(method='ffill', inplace=True)

    return df


def get_datasets():
    '''
    Return training and validation set
    '''
    training = csv_to_dataframe(TRAINING_DATA)
    validation = training.iloc[6000:]
    training = training.iloc[0: 6000]
    test = csv_to_dataframe(TEST_DATA)
    return (LandmarkLoader(training),
            LandmarkLoader(validation),
            LandmarkLoader(test))


class LandmarkLoader(Dataset):
    '''
    Returns the features and labels for the given dataframe
    '''
    def __init__(self, dataframe, transforms=transforms.ToTensor()):
        '''
        Initialize with the correct dataframe
        '''
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        '''
        Get length of dataset
        '''
        return len(self.dataframe)

    def __getitem__(self, index):
        '''
        Get one sample
        '''
        row = self.dataframe.iloc[index]
        feature = np.array([int(num) for num in row[-1].split(' ')],
                           dtype=np.uint8)
        feature = np.reshape(feature, INPUT_DIMS)
        label = np.array(row[:-1], dtype=np.float32)
        if np.isnan(label).any():
            import pdb; pdb.set_trace()
            print('got a nan')
        if self.transforms:
            feature = self.transforms(feature)

        return feature, label


if __name__ == '__main__':
    train, validation, test = get_datasets()
    print('done')