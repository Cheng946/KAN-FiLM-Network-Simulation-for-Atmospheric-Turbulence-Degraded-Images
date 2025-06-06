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



def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def test(net, test_iter, criterion,lr_scheduler, device):
    total, correct,test_loss = 0,0,0
    #将模型设置为测试模式
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            # y = np.expand_dims(y, axis=1)
            # y = torch.from_numpy(y)

            #用模型估算输出的结果
            output = net(X.cuda())


            loss = criterion(output, y.cuda())

            test_loss += loss.item()
            total += 1

    # 计算得到损失函数的均值
    test_loss_mean = test_loss / total

    if lr_scheduler is not None:
        lr_scheduler.step(test_loss_mean)
    print("test_loss_mean: {:.6f}"\
          .format(test_loss_mean))
    print("************************************\n")
    net.train()

    return test_loss_mean


def train(net, train_iter, criterion, optimizer, num_epochs, num_print, lr_scheduler=None, test_iter=None):

    # a) model.train() ：启用BatchNormalization和Dropout。
    # 在模型测试阶段使用model.train()让model变成训练模式，此时dropout和batchnormalization的操作在训练q起到防止网络过拟合的问题。
    # b) model.eval()，不启用BatchNormalization和Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    # 不然的话，一旦test的batch_size过小，很容易就会因BN层导致模型performance损失较大；

    net.train()
    record_train = list()
    record_test = list()

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        total, correct, train_loss = 0, 0, 0
        #获取开始的时间
        start = time.time()

        # torch.nn.utils.clip_grad_norm(net.parameters(),max_norm=1.0)
        #开始训练，i=训练集图片数量/batch_size
        for i, (X, y) in enumerate(train_iter):
            #这里y同样是图片，所以不需要进行升维
            # y =np.expand_dims(y,axis=1)
            # y=torch.from_numpy(y)
            #这里更改了打印参数，设置为小数点后20位
            torch.set_printoptions(precision=20)
            # print(y[0][0])

            #将batch_size张训练图片放入模型中
            output = net(X.cuda())


            #计算损失值
            loss = criterion(output, y.cuda())

            #根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
            #但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad了。
            optimizer.zero_grad()
            #将损失函数反向传播
            loss.backward()
            #更新优化器
            optimizer.step()

            #将一个batch_size的损失加起来
            train_loss += loss.item()
            #将训练集的batch_size加起来
            # total += y.size(0)
            total += 1

            # correct += (output.argmax(dim=1) == y).sum().item()
            #计算得到损失函数的均值
            train_loss_mean = train_loss / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}]| train_loss_mean: {:.6f} | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), train_loss_mean, \
                         get_cur_lr(optimizer)))


        #打印一轮消耗的时间
        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        #如果测试加载器不为空，并记录测试和训练的均方差
        if test_iter is not None:
            record_test.append(test(net, test_iter, criterion,lr_scheduler, device))
        record_train.append(train_loss_mean)

        #保存最近的参数模型
        torch.save(net.module.state_dict(), opt.SaveLastPara)

    return record_train, record_test

def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train_loss_mean")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="val_loss_mean")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.ylim(0,np.min(record_train)*10)
    plt.xlabel("epoch")
    plt.ylabel("loss_mean")
    plt.ylim(ymin=0, ymax=1)

    plt.show()

def SaveLoss(Path,record):
    f=open(Path,"w")
    for line in record:
        f.write(str(line)+"\n")

    f.close()

parser = argparse.ArgumentParser()

parser.add_argument('--data_root',type=str,default="/media/aiofm/F/20240830_50000_A_OTF_PCA",help='数据集的根目录')
parser.add_argument('--input_nc', type=int, default=33, help='输入维度的通道数量')
parser.add_argument('--output_nc', type=int, default=70, help='输出维度的通道数量')
parser.add_argument('--batchSize', type=int, default=180, help='一次训练载入的数据量')
parser.add_argument('--learn_rate', type=float, default=0.00005, help='初始学习率')
parser.add_argument('--num_epochs', type=int, default=70, help='训练的轮数')


# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/Kan_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KanParam/Last_Kan_Para.pt", help='File path for saving the latest network parameters')

# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/Kan_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithAttention/Last_Kan_Para.pt", help='File path for saving the latest network parameters')

# parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/record_train_loss.txt", help='File path for saving training loss')
# parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/record_val_loss.txt", help='File path for saving evaluation loss')
# parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/MLP_Para.pt", help='File path for saving network parameters')
# parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_DnseKANParam/Last_MLP_Para.pt", help='File path for saving the latest network parameters')

parser.add_argument('--SaveTrainLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/record_train_loss.txt", help='File path for saving training loss')
parser.add_argument('--SaveValLossPath', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/record_val_loss.txt", help='File path for saving evaluation loss')
parser.add_argument('--SavePara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/Kan_Para.pt", help='File path for saving network parameters')
parser.add_argument('--SaveLastPara', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KANWithFiLM/Last_Kan_Para.pt", help='File path for saving the latest network parameters')

parser.add_argument('--num_print', type=int, default=10, help='每过num_print轮训练打印一次损失')

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

opt = parser.parse_args()

if opt.local_rank != -1:
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')


def main():
    # Initialize the neural network and instantiate the model

    # net=E_kan.KAN([opt.input_nc,64,128,64,opt.output_nc], grid_size=20, spline_order=70)
    # net=E_kan.DenseKAN([opt.input_nc,64,128,64,opt.output_nc], grid_size=20, spline_order=70)
    # net=E_kan.KANWithAttention([opt.input_nc,64,128,64,opt.output_nc], grid_size=20, spline_order=70)
    net=E_kan.KANWithFiLM([opt.input_nc,64,128,64,opt.output_nc], grid_size=20, spline_order=70,scale_noise=0.01,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1])

    print(net)
    print(sum(p.numel() for p in net.parameters()))

    # Load the model into the GPU
    net.to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        net = nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank)

    # The return values are three data loaders
    train_iter, test_iter, val_iter = MyKANnetLoader.load_dataset(opt)

    # Select the L1 loss function
    criterion =torch.nn.L1Loss()

    optimizer = optim.Adam(
        net.parameters(),
        lr=opt.learn_rate
    )

    # Set up learning rate decay: if the loss function does not decrease for 5 consecutive epochs,
    # decay the learning rate to 0.1 times its original value.

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=5,
                                                        verbose=True)

    record_train, record_val = train(net, train_iter, criterion, optimizer, \
                                     opt.num_epochs, opt.num_print, lr_scheduler, val_iter)

    # Set up storage files for record_train and record_val.
    SaveLoss(opt.SaveTrainLossPath, record_train)
    SaveLoss(opt.SaveValLossPath, record_val)

    # Save model parameters
    torch.save(net.module.state_dict(), opt.SavePara)  # The file extension is typically written as: .pt or .pth

    learning_curve(record_train, record_val)




if __name__ == '__main__':

    torch.cuda.empty_cache()
    main()