import os
srcDir = "/home/akunchala/Documents/z_Datasets/MARS_Dataset/bbox_train"

for d in os.listdir(srcDir) :
    os.makedirs("mars_derived",exist_ok=True)
    cmdToRun = F"python pipeline.py --src_imgs {os.path.join(srcDir,d)} --tmp_dir {os.path.join('mars_derived',d)} --func mars"
    os.system(cmdToRun)