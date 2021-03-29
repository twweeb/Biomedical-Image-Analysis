import pydicom
import time
import os
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sqrt

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
    dicom_list.sort(key=lambda x: x.InstanceNumber, reverse=True)
    slice_thickness = abs(dicom_list[0].ImagePositionPatient[2] - dicom_list[1].ImagePositionPatient[2])
    for slice in dicom_list:
        slice.SliceThickness = slice_thickness

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
    exclude_part = flatten_img.min()
    exclude_idx = flatten_img==exclude_part
    clean_img = flatten_img[~exclude_idx]
    median = np.median(clean_img)
    return median


def mean_threshold(img):
    flatten_img = img.flatten()
    exclude_part = flatten_img.min()
    exclude_idx = flatten_img==exclude_part
    clean_img = flatten_img[~exclude_idx]
    mean = clean_img.mean()
    return mean


def thresholding(img, method, threshold=None):
    if threshold == None:
        if method == 'mean':
            threshold = mean_threshold(img)
        elif method == 'median':
            threshold = median_threshold(img)
        else:
            raise ValueError
    white = img>=threshold
    black = img<threshold
    max_val = img.max()
    min_val = img.min()
    img[white] = max_val
    img[black] = min_val
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
            plt.imshow(thresholding(normal_img, method='mean'), cmap='gray')
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
    return 1 - imgs


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.axis('off')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [255.0/255.0, 54.0/255.0, 57/255.0]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


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

        # Plot Patient's CT in 3D resampling
        # patient_pixels = get_pixels_hu(patient_dicoms)
        # patient_pixels_resampled, spacing = resample(patient_pixels, patient_dicoms, [1,1,1])
        # print("Shape before resampling\t", patient_pixels.shape)
        # print("Shape after resampling\t", patient_pixels_resampled.shape)
        # segmented_lungs = segment_lung_mask(patient_pixels_resampled, False)
        # plot_3d(segmented_lungs, 0)

        break # demo only one patient