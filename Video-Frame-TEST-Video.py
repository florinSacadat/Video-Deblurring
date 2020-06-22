import cv2
import os
import glob
import os
import torch
from torch.utils.data import DataLoader
import DeBlurNet_Deep as M
import DatasetTEST as D
import VisdomUtils
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import utils.Metrics as metrics
import Frames2Video
import time

def Vid2Frames_Index(vid_link):

    out_link = vid_link[:-4] + "/InputFrames/"
    if not os.path.isdir(out_link):
        os.makedirs(out_link)
    file = open(out_link + "5Frames.txt", "w+")
    vidcap = cv2.VideoCapture(vid_link)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite(out_link+"/%s.png" % count.__str__().zfill(4),
                    image)  # save frame as png file
        success, image = vidcap.read()

        count += 1

    imList = glob.glob(out_link + '*.png')
    j = 1
    while j + 5 < len(imList):
        x1 = j + 1
        x2 = j + 2
        x3 = j + 3
        x4 = j + 4
        x5 = j + 5
        file.write(
           out_link+ '%04d.png' % x1 + ' , ' +    out_link + '%04d.png' % x2 + ' , ' +    out_link + '%04d.png' % x3 + ' , ' + out_link + '%04d.png' % x4 + ' , ' +   out_link + '%04d.png' % x5 + '\n')
        j = j + 1


    file.close()

    return out_link+"5Frames.txt"


def test_function(model, test_loader):

    psnr=0
    model.eval()

    for batch_idx, (inputs) in enumerate(test_loader):
        inputs = inputs.cuda()


        in_img = inputs
        with torch.no_grad():
         out_img = model(in_img)


        if not os.path.isdir(result_dir ):
            os.makedirs(result_dir)

        out_img=transforms.ToPILImage()(out_img['output1'][0].cpu())
        # in_img[2].save(result_dir+'FlorinDS_MSE_FS_DBN-MSE_00_in_%d.jpg' % batch_idx)
        out_img.save(result_dir +'%04d.jpg' % batch_idx)
        # target_img.save(result_dir +'FlorinDS_MSE_FS_DBN-MSE_00_target_%d.jpg' %  batch_idx)
        # psnr = metrics.PSNR(target_img, out_img)
        # print("PSNR image%d = %d ",(batch_idx,psnr))

    return 0

def make_HD(vid_link):
    format = 1080
    if format == 1080:
        converter = os.system(
            'ffmpeg -i ' + vid_link + ' -vf scale=-1:720 -c:v libx264 -crf 18 -preset veryslow -c:a copy ' + vid_link[
                                                                                                             :-4] + "HD.mp4")
    #vid_link = vid_link[:-4] + "HD.mp4"
    return vid_link

if __name__ == '__main__':

    # vid_link = '/media/student/2.0 TB Hard Disk/VideoTesting/TestDS/%04d/'
    # for i in range(5,8):
    #     vid_link = '/media/student/2.0 TB Hard Disk/VideoTesting/TestDS/%04d/Blur30fps.mp4'%i
    #
    # vid_link = '/media/student/2.0 TB Hard Disk/Final5_Tests/#1/blur/'
    to = time.time()
    vid_link = '/home/student/Documents/FlaskWebPlatform/FiXiT/VideosInput/2.mp4'

    # Frames2Video.Frames2Video(vid_link,vid_link)
        # control param
    is_video=True
    is_fHD=False
    if is_video:
        if is_fHD:
            vid_link= make_HD(vid_link)


    print (Vid2Frames_Index(vid_link))

    file_root_blur=Vid2Frames_Index(vid_link)
    # file_root_blur='/media/student/2.0 TB Hard Disk/Final5_Tests/RealVideo/PapreCompare/RealVidBook/b.txt'
    # vid_link='/media/student/2.0 TB Hard Disk/Final5_Tests/RealVideo/PapreCompare/RealVidBook/Results/'
    result_dir=vid_link[:-4]+"Results/"
    # result_dir=vid_link
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    model_dir="/media/student/2.0 TB Hard Disk/modelsCkp/DBN_D/FlorinDS_FS_MSE_20/"
    model_name="DBN_D_20_FlorinDS_MSE_FS_Checkpoint_e0345"

    test_set = D.FlorinDataset(file_root_blur,transforms)
    test_loader = DataLoader(test_set, batch_size=1,shuffle=False,num_workers=6)

    model=M.DeBlurNet().cuda()
    model.load_state_dict(torch.load(model_dir+model_name))
    test_function(model,test_loader)

    result_name=result_dir+"Result.mp4"
    Frames2Video.Frames2Video(result_dir,result_name)
    make_HD(result_name)
    t1 = time.time()
    total = t1 - to
    print(total)

