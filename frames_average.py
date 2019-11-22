import glob
import os, numpy, PIL
from PIL import Image
import  shutil
# Access all PNG files in directory

w=1280
h=720
average_lenght=7

for i in range(3,4):
    in_path='/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Original/'+i.__str__()+'/'

    if not os.path.exists('/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Blur/' + i.__str__() + '/'):
        os.makedirs('/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Blur/' + i.__str__() + '/')

    if not os.path.exists('/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Sharp/' + i.__str__() + '/'):
        os.makedirs('/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Sharp/' + i.__str__() + '/')

    blur_out_path = '/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Blur/' + i.__str__() + '/'
    sharp_out_path = '/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Sharp/' + i.__str__() + '/'
    imlist = [f for f in glob.glob(in_path+"*.png", recursive=True)]
    # imlist=os.listdir(in_path)
    imlist.sort()
    print(imlist)
    N = len(imlist)
    name_nr = 0
    imarr = numpy.array([0])
    for i in range(0, N, average_lenght):
        arr = numpy.zeros((h, w, 3), numpy.float)
        name_nr+=1

        for im in imlist[i:i+average_lenght]:
            if(i+int((average_lenght/2))<N):
                shutil.copy(imlist[i+int((average_lenght/2))],sharp_out_path)
            imarr = numpy.array(Image.open(im), dtype=numpy.float)
            arr = arr + imarr / average_lenght
        # Round values in array and cast as 8-bit integer
        arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)
        # Generate, save and preview final image
        out = Image.fromarray(arr, mode="RGB")
        out.save(blur_out_path+name_nr.__str__().zfill(4)+".png")

for f in range(3,4):
  path='/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Sharp/'+f.__str__()+'/'
  all_files = os.listdir(path)
  all_files.sort()
  print(all_files)
  for i in range(1,len(all_files)):
        os.rename(path+all_files[i],path+i.__str__().zfill(4)+'.png')

