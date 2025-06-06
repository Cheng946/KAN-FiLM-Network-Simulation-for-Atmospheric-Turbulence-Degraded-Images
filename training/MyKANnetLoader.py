
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

import numpy as np

def numpy_to_tensor(np_array):
    return torch.from_numpy(np_array).float()

# Define and load the dataset
def load_dataset(opt):
    # Create the dataset using your custom MyDataset class! Note: It's the dataset (not the DataLoader iterator) that needs to be instantiated.

    # Dataset loading method configuration
    train_data = MyDataset(inputPath=opt.data_root+'/Train/Train_input.npy',LabelPath=opt.data_root+'/Train/Train.npy', transform=numpy_to_tensor)
    test_data = MyDataset(inputPath=opt.data_root+'/Test/Test_input.npy',LabelPath=opt.data_root+'/Test/Test.npy', transform=numpy_to_tensor)
    val_data = MyDataset(inputPath=opt.data_root+'/Val/Val_input.npy',LabelPath=opt.data_root+'/Val/Val.npy', transform=numpy_to_tensor)

    # Define a sampler
    train_sampler = DistributedSampler(train_data)
    test_sampler= DistributedSampler(test_data,shuffle=False)
    val_sampler= DistributedSampler(val_data)

    # Next, invoke DataLoader along with the dataset created earlier to build the dataloader.
    # It's worth noting that the length of the dataloader corresponds to the number of batches,
    # which is why it is related to the batch_size.
    train_loader = DataLoader(dataset=train_data,sampler=train_sampler, batch_size=opt.batchSize, num_workers=12,pin_memory=True)
    test_loader = DataLoader(dataset=test_data,sampler=test_sampler, batch_size=opt.batchSize, num_workers=12,pin_memory=True)
    val_loader = DataLoader(dataset=val_data,sampler=val_sampler, batch_size=opt.batchSize, num_workers=12,pin_memory=True)

    return train_loader, test_loader, val_loader

# *************************************Dataset configuration****************************************************************************
# Define the file reading format

class MyDataset(Dataset):
    # Create your own class MyDataset, which inherits from torch.utils.data.Dataset.
    # ********************************** Use __init__() to initialize necessary parameters and invoke the dataset **********************
    def __init__(self, inputPath ,LabelPath , transform=None):
        super(MyDataset, self).__init__()
        # Initialize the attributes inherited from the parent class
        self.input = np.load(inputPath,allow_pickle=True)
        self.input=np.array(self.input,dtype=np.float64)

        self.label=np.load(LabelPath,allow_pickle=True)
        self.label=np.array(self.label,dtype=np.float64)

        self.transform = transform
        # *************************** Use __getitem__() to preprocess the data and return the desired information**********************

    def __getitem__(self, index):
        PCA_input = self.input[index]
        PCA_label = self.label[index]

        # Change numpy to tensor
        if PCA_label is not None:
            PCA_input = self.transform(PCA_input)
            PCA_label = self.transform(PCA_label)

            PCA_input=PCA_input.unsqueeze(0)
            PCA_label=PCA_label.unsqueeze(0)

        return PCA_input,PCA_label

    # ********************************** Use __len__() to initialize parameters that need to be passed in and invoke the dataset **********************

    def __len__(self):
        # This method must also be implemented, as it returns the length of the dataset.
        return self.input.shape[0]
