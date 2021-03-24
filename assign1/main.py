import pydicom
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
from pathlib import Path
from pydicom.data import get_testdata_file
from pydicom.waveforms import generate_multiplex
from pydicom import dcmread

dataset_dir = '../CT_chest_scans/'
numpy_format_dir = '../CT_chest_scans_numpy/'


def read_by_patient_id(patient_id):
    data_path = dataset_dir + str(patient_id) + '/'
    patient_dcms = [data_path + img_name for img_name in [x for x in os.walk(data_path)][0][2]]

    return patient_dcms


def print_data_fields(dsm_img):
    # dsm_img = get_testdata_file(filename)
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

def iterate_patient_records(patient_dir):
    path = get_testdata_file(patient_dir)
    ds = dcmread(patient_dir)
    root_dir = Path(ds.filename).resolve().parent
    print(f'Root directory: {root_dir}\n')

    for patient in ds.patient_records:
        print(
            f"PATIENT: PatientID={patient.PatientID}, "
            f"PatientName={patient.PatientName}"
        )

        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            descr = study.StudyDescription or "(no value available)"
            print(
                f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                f"StudyDate={study.StudyDate}, StudyDescription={descr}"
            )

            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            for series in all_series:
                # Find all the IMAGE records in the series
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]
                plural = ('', 's')[len(images) > 1]

                descr = getattr(
                    series, "SeriesDescription", "(no value available)"
                )
                print(
                    f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                    f"Modality={series.Modality}, SeriesDescription={descr} - "
                    f"{len(images)} SOP Instance{plural}"
                )

                # Get the absolute file path to each instance
                #   Each IMAGE contains a relative file path to the root directory
                elems = [ii["ReferencedFileID"] for ii in images]
                # Make sure the relative file path is always a list of str
                paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                paths = [Path(*p) for p in paths]

                # List the instance file paths
                for p in paths:
                    print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

                    # Optionally read the corresponding SOP Instance
                    # instance = dcmread(Path(root_dir) / p)
                    # print(instance.PatientName)

if __name__ == '__main__':

    patient_ids = [x[1] for x in os.walk(dataset_dir)][0]

    for i in range(len(patient_ids)):
        patient_cts = read_by_patient_id(patient_ids[i])

        # iterate_patient_records(patient_cts[0])
        print_data_fields(patient_cts[0])

        break
