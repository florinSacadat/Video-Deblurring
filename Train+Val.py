import os
import numpy as np
import glob
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import DeBlurNet as M
import DatasetCrop as DC
import DatasetFull as DF
import VisdomUtils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import utils.Metrics as metrics
import time

# # Train paths and Load

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


def valid_loss_function(model, val_loader,epoch,save_freq):
    psnr=0
    final_loss=0
    g_loss = np.zeros((5000, 1))
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs, targets


        in_img = inputs
        target = targets

        out_img = model(in_img)
        out_img = out_img

        loss = reduce_mean(out_img['output1'], target)

        g_loss[batch_idx] = loss.data.cpu()
        final_loss = np.mean(g_loss[np.where(g_loss)])
        print("%d %d Loss=%.10f" % (epoch, batch_idx, final_loss))
    if epoch % save_freq == 0:
        if not os.path.isdir(result_dir_val + '%04d' % epoch):
            os.makedirs(result_dir_val + '%04d' % epoch)

        out_img=transforms.ToPILImage()(out_img['output1'][0].cpu())
        target_img = transforms.ToPILImage()(target[0].cpu())
        out_img.save(result_dir_val +'/%04d/' % epoch+ '%04dDBN-FlorinDS_MSE_FS_30_00_train_%d.jpg' % (epoch, batch_idx))
        target_img.save(result_dir_val +'/%04d/' % epoch+ '%04dDBN-FlorinDS_MSE_FS_30_00_target_%d.jpg' % (epoch, batch_idx))
    psnr = metrics.PSNR(target_img, out_img)

    return final_loss, psnr


if __name__ == '__main__':
    file_root_blur='/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nTRAIN/text/2-31B.txt'
    file_root_sharp='/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nTRAIN/text/2-31S.txt'
    file_root_blurVal = '/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nVALIDATION/text/all_V_blur.txt'
    file_root_sharpVal = '//media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nVALIDATION/text/all_V_sharp.txt'
    result_dir_train="/media/student/2.0 TB Hard Disk/Results/FlorinDS_MSE_FS_30/Train/"
    result_dir_val="/media/student/2.0 TB Hard Disk/Results/FlorinDS_MSE_FS_30/Validation/"
    model_dir="/media/student/2.0 TB Hard Disk/modelsCkp/FlorinDS_MSE_FS_30/"
    # model_name="FlorinDS_MSE_FS_30"

    train_set = DF.FlorinDataset(file_root_blur,file_root_sharp,transforms)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False,num_workers=6)

    val_set = DF.FlorinDataset(file_root_blurVal,file_root_sharpVal,transforms)
    val_loader = DataLoader(val_set, batch_size=1,shuffle=False,num_workers=6)

    g_loss = np.zeros((5000, 1))

    save_freq_Img=1
    save_freq_model=5
    learning_rate = 1e-4
    model=M.DeBlurNet().cuda()
    # model.load_state_dict(torch.load(model_dir+model_name))
    s_epoch=0
    # s_epoch=int(model_name[-3:])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    plotter1 = VisdomUtils.VisdomLinePlotter(env_name='FlorinDS_MSE_FS_30_Train Loss')
    plotter2 = VisdomUtils.VisdomLinePlotter(env_name='FlorinDS_MSE_FS_30_Train PSNR')
    plotter1V = VisdomUtils.VisdomLinePlotter(env_name='FlorinDS_MSE_FS_30_Validation Loss')
    plotter2V = VisdomUtils.VisdomLinePlotter(env_name='FlorinDS_MSE_FS_30_Validation PSNR')


    final_loss = 0
    save_freq=5
    criterion = torch.nn.MSELoss()
    # criterion = reduce_mean()
    psnr=0
    for epoch in range(s_epoch,501):
        if s_epoch>200:
            learning_rate=1e-5
        to=time.time()

        model.train()
        print('Epoch:' + str(epoch))
        if os.path.isdir("result/%04d" % epoch):
            continue

        for batch_idx, (input, target) in enumerate(train_loader):
            if torch.cuda.is_available():
               input,target=input.cuda(), target.cuda()


            optimizer.zero_grad()
            outputs = model(input)

            loss= criterion(outputs['output1'],target)
            # loss=Loss.DBNLoss.forward(outputs,target)
            loss.backward()

            optimizer.step()
            g_loss[batch_idx] = loss.data.cpu()
            final_loss = np.mean(g_loss[np.where(g_loss)])

            print("%d %d Loss=%.10f" % (epoch, batch_idx, final_loss))
        if epoch % save_freq_Img == 0:
            if not os.path.isdir(result_dir_train + '%04d' % epoch):
                os.makedirs(result_dir_train + '%04d' % epoch)

            out_img=transforms.ToPILImage()(outputs['output1'][0].cpu())
            target_img = transforms.ToPILImage()(target[0].cpu())
            out_img.save(result_dir_train +'/%04d/' % epoch+ '%04dFlorinDS_MSE_FS_30_00_train_%d.jpg' % (epoch, batch_idx))
            target_img.save(result_dir_train +'/%04d/' % epoch+ '%04dFlorinDS_MSE_FS_30_00_target_%d.jpg' % (epoch, batch_idx))
            if epoch % save_freq_model == 0:  torch.save(model.state_dict(), model_dir + 'FlorinDS_MSE_FS_30_Checkpoint_e%04d' % epoch)
            model_name='FlorinDS_MSE_FS_30_Checkpoint_e%04d' % epoch

            if epoch % save_freq == 0 and batch_idx == (len(train_loader) - 1):
                model_V= M.DeBlurNet().cpu()
                model_V.load_state_dict(torch.load(model_dir + model_name))
                my_val_loss, val_psnr = valid_loss_function(model_V.cpu(), val_loader, epoch, save_freq)
                print("Validation Loss=%.10f" % my_val_loss)
                plotter1V.plot('loss', 'Validation Loss', 'Epoch', epoch, my_val_loss)
                print("PSNR: " + str(val_psnr))
                plotter2V.plot("PSNR", 'Val_PSNR', 'Epoch', epoch, val_psnr)
            psnr=metrics.PSNR(target_img,out_img)
            print("PSNR: " + str(psnr))
        # ssim=metrics.SSIM(target_img,out_img)
        plotter1.plot('Loss', 'Train Loss', 'Epoch', epoch, final_loss)
        plotter2.plot("PSNR", 'Train_PSNR', 'Epoch', epoch, psnr)
        t1=time.time()
        total=t1-to
        print(total)
        # torch.save(model.state_dict(), model_dir + 'checkpoint_curr_e%04d' % epoch)












