
from src import model as E_kan

import torch
import torch.nn as nn
import MyKANnetLoader
import argparse
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import joblib
from sklearn.decomposition import PCA
from scipy.signal import correlate
import time

# This file is used to test the output accuracy of the trained network
parser = argparse.ArgumentParser()

parser.add_argument('--data_root',type=str,default="/media/aiofm/F/20240830_50000_A_OTF_PCA",help='The root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=33, help='The number of channels in the input dimension')
parser.add_argument('--output_nc', type=int, default=70, help='The number of channels in the output dimension')
parser.add_argument('--batchSize', type=int, default=262144, help='The amount of data loaded in one training session')
parser.add_argument('--learn_rate', type=float, default=0.00005, help='Initial learning rate')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')

# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/Kan_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/Last_Kan_Para.pt", help='File path for saving the latest network parameters')


# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/MLP_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/Last_MLP_Para.pt", help='File path for saving the latest network parameters')

# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/Kan_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/Last_Kan_Para.pt", help='File path for saving the latest network parameters')

parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/record_train_loss.txt", help='File path for saving training loss')
parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/record_val_loss.txt", help='File path for saving evaluation loss')
parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/Kan_Para.pt", help='File path for saving network parameters')
parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/Last_Kan_Para.pt", help='File path for saving the latest network parameters')


parser.add_argument('--num_print', type=int, default=10, help='Print the loss every num_print training epochs')

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

opt = parser.parse_args()

if opt.local_rank != -1:
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')



# Test the network
def test(test_loader,model,criterion):
    # Set the model to evaluation mode
    model.eval()
    # Do not perform gradient backpropagation
    with torch.no_grad():
        i = 0;
        for data in test_loader:
            inputs, label  = data
            inputs, label = inputs.to(device), label.to(device)

            start_time=time.time()
            outputs = model(inputs)
            end_time=time.time()

            print(end_time-start_time)

            # Select a OTF index.
            matrix_size=128
            outputs1 = outputs[0].cuda().data.cpu().numpy()
            # print(outputs1.shape)
            label1=label[0].cuda().data.cpu().numpy()

            # Compare the basis function coefficients of the outputs

            # plt.plot(range(outputs1.shape[1]), outputs1[0,:], color='blue', marker='o', linestyle='-', label='output')
            # plt.plot(range(label1.shape[1]), label1[0,:], color='red', marker='x', linestyle='-', label='label')
            #
            # plt.ylim(-2.5,2.5)
            #
            # plt.legend()
            # plt.show()

            pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/pca_model-70.pkl')
            reduced_matrices = pca.inverse_transform(outputs1)
            approx_real_part = reduced_matrices[:, :matrix_size * matrix_size].reshape(matrix_size,
                                                                                                matrix_size)
            approx_imag_part = reduced_matrices[:, matrix_size * matrix_size:].reshape(matrix_size,
                                                                                                matrix_size)
            # Decompose the reconstructed matrix into real and imaginary parts,
            # and reconstruct the complex matrix.

            approx_complex_matrix = approx_real_part + 1j * approx_imag_part



            label_reduced=pca.inverse_transform(label1)
            label_approx_real_part = label_reduced[:, :matrix_size * matrix_size].reshape(matrix_size,
                                                                                       matrix_size)
            label_approx_imag_part = label_reduced[:, matrix_size * matrix_size:].reshape(matrix_size,
                                                                                       matrix_size)
            # Reconstruct the matrix using the ground truth label.
            label_approx_complex_matrix = label_approx_real_part + 1j * label_approx_imag_part

            print(approx_complex_matrix[64,64].real,label_approx_complex_matrix[64,64].real,approx_complex_matrix[64,64].real/label_approx_complex_matrix[64,64].real*7.94)

            # Visual comparison

            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # axs[0].imshow(np.abs(approx_complex_matrix ))
            # axs[0].set_title('Reconstruct the OTF using the network outputs')
            # axs[0].axis('off')
            #
            # axs[1].imshow(np.abs(label_approx_complex_matrix))
            # axs[1].set_title('Reconstruct the OTF using the label values')
            # axs[1].axis('off')
            #
            # plt.show()

        print("finished")




if __name__ == '__main__':
    # Free up GPU memory
    torch.cuda.empty_cache()

    # Initialize the neural network and instantiate the model

    # net = E_kan.KAN([opt.input_nc,64,128,64,opt.output_nc], grid_size=20, spline_order=70)
    # net = E_kan.DenseKAN([opt.input_nc, 64, 128, 64, opt.output_nc], grid_size=20, spline_order=70)
    # net = E_kan.KANWithAttention([opt.input_nc, 64, 128, 64, opt.output_nc], grid_size=20, spline_order=70)
    net = E_kan.KANWithFiLM([opt.input_nc, 64, 128, 64, opt.output_nc], grid_size=20, spline_order=70, scale_noise=0.01,
                            scale_base=1.0,
                            scale_spline=1.0,
                            base_activation=torch.nn.SiLU,
                            grid_eps=0.02,
                            grid_range=[-1, 1])

    # Load the model into the GPU
    net = net.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        net = nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank)


    # Load the model parameters into the model
    net.module.load_state_dict(torch.load(opt.SavePara))

    # The return values are three data loaders
    train_iter, test_iter,val_test = MyKANnetLoader.load_dataset(opt)

    criterion = torch.nn.L1Loss()

    test(test_iter,net,criterion)