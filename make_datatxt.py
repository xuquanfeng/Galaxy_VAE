import os
from astropy.io import fits
from tqdm import tqdm
b = 0
dir = '/data/GZ_Decals/'
files = os.listdir(dir)
train = open('train.txt', 'w')#创建txt文件用于后续数据储存
for b in range(len(files)):
    #这里采用的是判断文件名的方式进行处理
    if 'merge' == files[b] or 'nomerge' == files[b]:
        ss = dir + str(files[b]) + '/'
        pics = os.listdir(ss)
        for i in tqdm(range(len(pics))):
            if 'fits' in pics[i]:
                try:
                    name = str(dir) + str(files[b]) + '/' + pics[i] + '\n'
                    hdu = fits.open(name[:-1])
                    img = hdu[0].data
                    if len(img)==3 and len(img[0])==256:
                        train.write(name)
                    hdu.close()
                except:
                    continue
train.close()