import pydicom
import numpy as np
import matplotlib.pyplot as plt
import skimage
from math import sqrt
import os

dataset_dir = '../CT_chest_scans/'
result_img_dir = 'CT_chest_scans_segmentation/'


def read_by_patient_id(patient_id):
    data_path = dataset_dir + str(patient_id) + '/'
    patient_dcms = [data_path + img_name for img_name in [x for x in os.walk(data_path)][0][2]]

    return patient_dcms


def print_data_fields(dsm_img):
    ds = pydicom.dcmread(dsm_img)

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

def extract_raw_and_hounsfield (dcm_img):
    ds = pydicom.dcmread(dcm_img)
    rSlope = ds.RescaleSlope
    rIntercept = ds.RescaleIntercept
    instanceNum = ds.InstanceNumber
    img_height, img_width = ds.pixel_array.shape
    dcm_img_pixel_array = np.clip(ds.pixel_array, 0, None)

    dcm_img_hu = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            dcm_img_hu[i][j] = dcm_img_pixel_array[i][j]*rSlope + rIntercept
    return instanceNum, dcm_img_pixel_array, dcm_img_hu


def thresholding(img):
    threshold = 0.24 #img.median()

    white = img>=threshold
    black = img<threshold
    img[white] = 1.
    img[black] = 0.
    return img


def img_nomarlize(img):
    img_max = img.max()
    img_min = img.min()

    normalized_img = (img - img_min) / (img_max - img_min)

    return normalized_img

def print_list_of_images(img_list, patient_id):
    h, w = img_list[0].shape
    fig = plt.figure(figsize=(10, 10))
    columns = rows = int(sqrt(len(img_list)))

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("slice:" + str(i))  # set title
        plt.imshow(thresholding(img_nomarlize(img_list[i])), cmap='gray')
        plt.axis('off')


    plt.show()  # finally, render the plot
    if not os.path.exists(result_img_dir):
        os.mkdir(result_img_dir)
    fig.savefig(result_img_dir + 'patient_' + str(patient_id))


def show_hist(img_pixel):
    plt.hist(img_pixel.flatten(), bins=50, color='c')
    plt.title('Normalized Image')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':

    patient_ids = [x[1] for x in os.walk(dataset_dir)][0]

    for i in range(len(patient_ids)):
        patient_cts = read_by_patient_id(patient_ids[i])

        hu_imgs = list()
        hu_imgs_id = list()
        cnt = 0

        for patient_ct in patient_cts:
            # Read in and print out all the data fields in a DICOM file
            # print(patient_ct)
            # print_data_fields(patient_ct)

            patient_ct_id, patient_ct_raw, patient_ct_hu = extract_raw_and_hounsfield (patient_ct)

            # Read in the raw data for a CT slice and convert its pixel values into Hounsfield units.
            # Compute the max, min, mean and standard deviation of both images (raw data and Hounsfield units).
            # raw_max, raw_min, raw_mean, raw_std = (patient_ct_raw.max(), patient_ct_raw.min(), patient_ct_raw.mean(), patient_ct_raw.std())
            # hu_max, hu_min, hu_mean, hu_std = (patient_ct_hu.max(), patient_ct_hu.min(), patient_ct_hu.mean(), patient_ct_hu.std())
            #
            # print(f'raw_max: {raw_max}, raw_min: {raw_min}, raw_mean: {raw_mean}, raw_std: {raw_std}')
            # print(f'hu_max: {hu_max}, hu_min: {hu_min}, hu_mean: {hu_mean}, hu_std: {hu_std}')

            # show_hist(normal_img)

            hu_imgs.append(patient_ct_hu)
            hu_imgs_id.append(patient_ct_id)
            cnt = cnt + 1
            if cnt == 25:
                break

        hu_imgs = [x for _,x in sorted(zip(hu_imgs_id,hu_imgs))]
        print_list_of_images(hu_imgs, i)
        break
