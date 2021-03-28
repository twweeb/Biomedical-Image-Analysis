import pydicom
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time
import os
from vedo import *

dataset_dir = '../CT_chest_scans/'
result_img_dir = 'CT_chest_scans_segmentation/'


def read_by_patient_id(patient_id):
    data_path = dataset_dir + str(patient_id) + '/'
    patient_dcms = [data_path + img_name for img_name in [x for x in os.walk(data_path)][0][2]]

    return patient_dcms


def print_data_fields(ds):
    # get the pixel information into a numpy array
    data = ds.pixel_array
    print('The image has {} x {} voxels'.format(data.shape[0],
                                                data.shape[1]))
    data_downsampling = data[::8, ::8]
    print('The downsampled image has {} x {} voxels'.format(
        data_downsampling.shape[0], data_downsampling.shape[1]))

    # copy the data back to the original data set
    ds.PixelData = data_downsampling.tobytes()
    # update the information regarding the shape of the data array
    ds.Rows, ds.Columns = data_downsampling.shape

    # print the image information given in the dataset
    print('The information of the data set after downsampling: \n')
    print(ds)


def compute_statistic(img):
    flatten_img = img.flatten()
    zero_idx = flatten_img==flatten_img.min()
    res = flatten_img[~zero_idx]

    return res.max(), res.min(), res.mean(), res.std()

def read_dicom_to_list (patient_dicoms):
    st = time.time()
    dicom_list = list()
    dicom_num = 0
    dicom_reject = 0
    for file_name in patient_dicoms:
        ds = pydicom.dcmread(file_name)
        try:
            if hasattr(ds, 'SliceLocation'):
                dicom_list.append(ds)
                dicom_num += 1
            else:
                dicom_reject += 1
                print("reject: ", file_name)
        except Exception as e:
            pass

    print("read done %s s | %d files | %d rejected" % (round(time.time() - st,2), dicom_num, dicom_reject))

    if len(dicom_list) <= 0:
        return False
    dicom_list.sort(key=lambda x: x.InstanceNumber)

    return dicom_list


def get_pixel_array_from_dicom (dicom_list):
    dcm_img_raw_list = list()
    dcm_img_hu_list = list()

    for dicom_file in dicom_list:
        rSlope = dicom_file.RescaleSlope
        rIntercept = dicom_file.RescaleIntercept
        pix_spacing = dicom_file.PixelSpacing
        img_height, img_width = dicom_file.pixel_array.shape

        dcm_img_raw = np.clip(dicom_file.pixel_array.astype(np.float32), 0, None)
        dcm_img_hu = dcm_img_raw*rSlope + rIntercept

        dcm_img_raw_list.append(dcm_img_raw)
        dcm_img_hu_list.append(dcm_img_hu)

    return dcm_img_raw_list, dcm_img_hu_list

def median_threshold(img):
    flatten_img = img.flatten()
    zero_idx = flatten_img==0
    removed_zero = flatten_img[~zero_idx]
    median = np.median(removed_zero)
    return median

def mean_threshold(img):
    flatten_img = img.flatten()
    zero_idx = flatten_img==0
    removed_zero = flatten_img[~zero_idx]
    mean = removed_zero.mean()
    return mean

def thresholding(img, threshold):
    white = img>=threshold
    black = img<threshold
    img[white] = 1.
    img[black] = 0.
    return img


def img_normalize(img):
    img_max = img.max()
    img_min = img.min()

    normalized_img = (img - img_min) / (img_max - img_min)

    return normalized_img

def print_list_of_images(img_list, patient_id, apply_threshold=True, save_img=False):
    h, w = img_list[0].shape
    fig = plt.figure(figsize=(10, 10))
    columns = rows = int(sqrt(len(img_list)))

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("slice:" + str(i))  # set title
        normal_img = img_normalize(img_list[i])
        if apply_threshold:
            threshold = mean_threshold(normal_img)
            plt.imshow(thresholding(normal_img, threshold), cmap='gray')
        else:
            plt.imshow(normal_img, cmap='gray')
        plt.axis('off')


    plt.show()  # finally, render the plot
    if save_img:
        if not os.path.exists(result_img_dir):
            os.mkdir(result_img_dir)
        fig.savefig(result_img_dir + 'patient_' + format(patient_id, '02d'))


def print_one_image(img, img_name):
    h, w = img.shape
    fig = plt.figure(figsize=(8, 8))

    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.show()  # finally, render the plot
    if not os.path.exists(result_img_dir):
        os.mkdir(result_img_dir)
    fig.savefig(result_img_dir + img_name)

def show_hist(img_pixel, threshold_value=None):
    # remove value 0
    flatten_img = img_pixel.flatten()
    zero_idx = flatten_img==0
    plt.hist(flatten_img[~zero_idx], bins=50, color='c')
    if threshold_value:
        plt.axvspan(threshold_value, threshold_value+0.001, color='red', alpha=0.5)
    plt.title('Normalized Image')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()

def flip_black_and_white(imgs):
    white = (imgs == 1.)
    new_img = imgs
    new_img[white] = 0.
    new_img[~white] = 1.
    return new_img

def draw_in_3D (dicom_files, hu_imgs):
    unset = True
    hu_stack = None
    pix_spacing = None
    z_spacing = None
    segmentation_imgs = [thresholding(img_normalize(hu_img), mean_threshold(hu_img)) for hu_img in hu_imgs]
    for i in range(len(dicom_files)):
        arr = flip_black_and_white(segmentation_imgs[i])*255

        if unset:
            imShape = (arr.shape[0], arr.shape[1], len(dicom_files))
            hu_stack = np.zeros(imShape)
            pix_spacing = dicom_files[i].PixelSpacing
            dist = 0
            for j in range(2):
                cs = [float(q) for q in dicom_files[j].ImageOrientationPatient]
                ipp = [float(q) for q in dicom_files[j].ImagePositionPatient]
                parity = pow(-1, j)
                dist += parity*(cs[1]*cs[5] - cs[2]*cs[4])*ipp[0]
                dist += parity*(cs[2]*cs[3] - cs[0]*cs[5])*ipp[1]
                dist += parity*(cs[0]*cs[4] - cs[1]*cs[3])*ipp[2]
            z_spacing = abs(dist)
            unset = False
        hu_stack[:, :, i] = arr

    pix_spacing.append(z_spacing)
    vol = Volume(hu_stack, spacing=pix_spacing)
    show(vol)


if __name__ == '__main__':

    patient_ids = [x[1] for x in os.walk(dataset_dir)][0]

    for i in range(len(patient_ids)):
        patient_cts = read_by_patient_id(patient_ids[i])
        patient_dicoms = read_dicom_to_list(patient_cts)

        # Read in and print out all the data fields in a DICOM file
        # print_data_fields(patient_dicoms[0])

        raw_imgs, hu_imgs = get_pixel_array_from_dicom(patient_dicoms)

        # Read in the raw data for a CT slice and convert its pixel values into Hounsfield units.
        # Compute the max, min, mean and standard deviation of both images (raw data and Hounsfield units).
        # raw_max, raw_min, raw_mean, raw_std = compute_statistic(raw_imgs[0])
        # hu_max, hu_min, hu_mean, hu_std = compute_statistic(hu_imgs[0])
        #
        # print(f'raw_max: {raw_max}, raw_min: {raw_min}, raw_mean: {raw_mean}, raw_std: {raw_std}')
        # print(f'hu_max: {hu_max}, hu_min: {hu_min}, hu_mean: {hu_mean}, hu_std: {hu_std}')

        print_list_of_images(hu_imgs[:25], i, apply_threshold=False, save_img=False)
        # draw_in_3D(patient_dicoms, hu_imgs)
        break # demo only one patient