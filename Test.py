import os
import torch
from torch.utils.data import DataLoader
import DeBlurNet as M
import DatasetTEST as D
import VisdomUtils
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import utils.Metrics as metrics


def test_function(model, test_loader):
    psnr=0
    model.eval()

    for batch_idx, (inputs) in enumerate(test_loader):
        inputs = inputs


        in_img = inputs

        out_img = model(in_img)
        out_img = out_img


        if not os.path.isdir(result_dir ):
            os.makedirs(result_dir)

        out_img=transforms.ToPILImage()(out_img['output1'][0].cpu())
        # in_img[2].save(result_dir+'FlorinDS_MSE_FS_DBN-MSE_00_in_%d.jpg' % batch_idx)
        out_img.save(result_dir +'%04d.jpg' % batch_idx)
        # target_img.save(result_dir +'FlorinDS_MSE_FS_DBN-MSE_00_target_%d.jpg' %  batch_idx)
        # psnr = metrics.PSNR(target_img, out_img)
        # print("PSNR image%d = %d ",(batch_idx,psnr))

    return 0


if __name__ == '__main__':
    file_root_blur='/home/student/Documents/Florin/EDVR-master/datasets/REDS4/blur/B2/B2.txt'
    result_dir="/home/student/Documents/Florin/EDVR-master/datasets/REDS4/blur/RESULT_Florin/B2/"
    model_dir="/media/student/2.0 TB Hard Disk/modelsCkp/FlorinDS_MSE_FS/"
    model_name="FS_checkpoint_FlorinDS_MSE_DBN_e0260"

    test_set = D.FlorinDataset(file_root_blur,transforms)
    test_loader = DataLoader(test_set, batch_size=1,shuffle=False,num_workers=6)

    model=M.DeBlurNet()
    model.load_state_dict(torch.load(model_dir+model_name))
    test_function(model,test_loader)














