import os
import glob


for f1 in range(1,3):
    converter = os.system(
         'ffmpeg -i o'+f1.__str__()+'.mp4 -vf scale=-1:720 -c:v libx264 -crf 18 -preset veryslow -c:a copy r_o'+f1.__str__()+'.MP4')
    print(f1)



