import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
dicom = np.load('/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_second_stage/mr_train_1001.npz', allow_pickle=True)["vol"]
dicom = np.transpose(dicom, axes=(1,0,2))
mask = np.load('/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_second_stage/mr_train_1001.npz', allow_pickle=True)["mask"]
mask = np.transpose(mask, axes=(1,0,2))

# dicom_img = sitk.GetImageFromArray(dicom)
# dicom_img.SetSpacing((1.211, 1.211, 0.9))
# sitk.WriteImage(dicom_img, '/home/taiping-qu/code/mr_heart_seg_thin/example/data/image_mr_1007.nii.gz')
# mask_img = sitk.GetImageFromArray(mask)
# mask_img.SetSpacing((1.211, 1.211, 0.9))
# sitk.WriteImage(mask_img, '/home/taiping-qu/code/mr_heart_seg_thin/example/data/image_mr_1007_seg.nii.gz')
print(np.unique(mask))
plt.subplot(1,2,1)
plt.imshow(dicom[55,:])
plt.subplot(1,2,2)
plt.imshow(mask[55,:])

plt.show()