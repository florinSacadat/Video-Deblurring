import cv2
import os
import glob
import os
import torch
from torch.utils.data import DataLoader
import DeBlurNet as M
import DatasetFullValid as DF
import VisdomUtils
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import utils.Metrics as metrics
import UtilsImage
import csv

with open('8DBN_FD_20s_512.csv', 'w') as f:
    writer = csv.writer(f)

    def test_function(model, test_loader,epoch,save_freq, result_dir_val,experiment_name,plotter1,plotter2):

        out_imgs=[]
        target_imgs=[]
        input_imgs=[]
        batch_i=0
        psnrs= []
        ssims=[]
        average_psnr=0
        average_ssim=0
        final_loss=0
        model.eval()
        for batch_idx, (inputs, targets,input_refs) in enumerate(test_loader):
            inputs, targets,input_refs = inputs, targets,input_refs


            in_img = inputs
            target = targets
            input_ref= input_refs


            out_img = model(in_img)
            out_img = out_img

            out_imgs=transforms.ToPILImage()(out_img['output1'][0].cpu())
            target_imgs = transforms.ToPILImage()(target[0].cpu())
            input_imgs = transforms.ToPILImage()(input_ref[0].cpu())
            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir_val + '%04d' % epoch):
                    os.makedirs(result_dir_val + '%04d' % epoch)
                psnrs.append(metrics.PSNR(target_imgs, out_imgs))
                ssims.append(metrics.SSIM(target_imgs, out_imgs))

                # paralel_imgs = []
                if batch_idx% 40==0:
                    input_imgs.save(
                        result_dir_val + '/%04d/' % epoch + '%04dDBN_FD_20s_512_00_input_%d.jpg' % (epoch, batch_idx))
                    out_imgs.save(
                        result_dir_val + '/%04d/' % epoch + '%04dDBN_FD_20s_512_00_train_%d-PSNR-%f.jpg' % (
                        epoch, batch_idx,psnrs[batch_idx]))
                    target_imgs.save(
                        result_dir_val + '/%04d/' % epoch + '%04dDBN_FD_20s_512_00_target_%d.jpg' % (epoch, batch_idx))
                # paralel_imgs.append(input_imgs)
                # paralel_imgs.append(out_imgs)
                # paralel_imgs.append(target_imgs)
                # UtilsImage.uniImage(paralel_imgs).save(
                #     result_dir_val + '/%04d/' % epoch + '%04dDBN_D_ER_FDS_MSE_FS_Result_%d-PSNR-%d.jpg' % (
                #     epoch, batch_i, psnrs[batch_idx]))

        # if epoch % save_freq == 0:
        #     if not os.path.isdir(result_dir_val + '%04d' % epoch):
        #         os.makedirs(result_dir_val + '%04d' % epoch)

            # for batch_i in range(0,test_loader):
            #     psnrs[batch_i] = metrics.PSNR(target_imgs[batch_i], out_imgs[batch_i])
            #     ssims[batch_i] = metrics.SSIM(target_imgs[batch_i], out_imgs[batch_i])
            #     if batch_i % 20 ==0:
            #         paralel_imgs =[]
            #         input_imgs[batch_i].save(result_dir_val +'/%04d/' % epoch+ '%04dDBNP_D_ER_FDS_MSE_FS_00_input_%d.jpg' % (epoch, batch_i))
            #         out_imgs[batch_i].save(result_dir_val +'/%04d/' % epoch+ '%04dDBNP_D_ER_FDS_MSE_FS_00_train_%d-PSNR-%d.jpg' % (epoch, batch_i,psnrs[batch_i]))
            #         target_imgs[batch_i].save(result_dir_val +'/%04d/' % epoch+ '%04dDBN_D_ER_FDS_MSE_FS_00_target_%d.jpg' % (epoch, batch_i))
            #         paralel_imgs.append( input_imgs[batch_i]  )
            #         paralel_imgs.append (out_imgs[batch_i]  )
            #         paralel_imgs.append(target_imgs[batch_i]  )
            #         UtilsImage.uniImage(paralel_imgs).save(result_dir_val +'/%04d/' % epoch+ '%04dDBN_D_ER_FDS_MSE_FS_Result_%d-PSNR-%d.jpg' % (epoch, batch_i,psnrs[batch_i]))
            #         # wandb.log({ '%04dDBN_D_ER_FDS_MSE_FS_Result_%d.jpg' % (epoch, batch_i) : wandb.Image( Utils.uniImage(paralel_imgs))
            #         #             , "PSNR: " :psnrs[batch_i]
            #         #             })
        for i in range(0,psnrs.__len__()):
            average_psnr += psnrs[i]
            average_ssim += ssims[i]

        average_psnr=average_psnr/psnrs.__len__()
        average_ssim=average_ssim/ssims.__len__()
        print(epoch,average_psnr,average_ssim)
        writer.writerow([epoch, average_psnr, average_ssim])

        plotter1.plot("average_PSNR", 'Val-PSNR', "Epoch", epoch, average_psnr)
        plotter2.plot("average_SSIM", 'Val-SSIM', "Epoch", epoch, average_ssim)

    def checkpoint_run(model_dir,model_name, model_number,result_dir_val,experiment_name,file_root_blurVal,file_root_sharpVal,plotter1,plotter2):
        val_set = DF.FlorinDataset(file_root_blurVal, file_root_sharpVal, transforms)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)
        model = M.DeBlurNet()
        for i in range (5,model_number.__int__(),5):
            model_name_number=model_name+ "%04d"%i
            model_load=model_dir+model_name_number
            model.load_state_dict(torch.load(model_load))
            test_function(model,val_loader,i,5,result_dir_val,experiment_name,plotter1,plotter2)



    if __name__ == '__main__':
        # set the model to test
        experiment_name='8DBN_FD_20s_512'
        model_dir = "/media/student/2.0 TB Hard Disk/modelsCkp/DBN/FlorinDS_MSE_512/"
        model_name = "MSEcheckpoint_DBN_e"
        model_number=201

        plotter1 = VisdomUtils.VisdomLinePlotter(env_name=experiment_name + 'Validation PSNR')
        plotter2 = VisdomUtils.VisdomLinePlotter(env_name=experiment_name + 'Validation SSIM')



        #set what to test on
        file_root_blurVal = '/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nVALIDATION/text/all_V_blur.txt'
        file_root_sharpVal = '/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/nVALIDATION/text/all_V_sharp.txt'

        # where to put it
        result_dir_val ='/media/student/2.0 TB Hard Disk/Final Graphs/'+experiment_name+'/'
        if not os.path.isdir(result_dir_val):
            os.makedirs(result_dir_val)

        checkpoint_run(model_dir,model_name,model_number,result_dir_val,experiment_name,file_root_blurVal,file_root_sharpVal,plotter1,plotter2)

