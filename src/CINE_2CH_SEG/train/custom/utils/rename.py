import os
import sys
def rename():
    target_path = '/home/qutaiping/nas/ori_data/AZHC/dul/del/4ch/part1/'
    fileType='_seg.nii.gz'
    count=0
    filelist=os.listdir(target_path)
    for files in filelist:
        Olddir=os.path.join(target_path, files)
        if os.path.isdir(Olddir):
            continue
        # name_list = 'Arterial' + files.split('.')[0][2:]
        # name_list = files.split('.')[0][:-17] + '-merged_mask'
        name_list = files.split('.nii.gz')[0][:5]
        # print(name_list)
        Newdir = os.path.join(target_path, name_list+fileType)
        os.rename(Olddir, Newdir)
        count+=1
    print("一共修改了"+str(count)+"个文件")
rename()