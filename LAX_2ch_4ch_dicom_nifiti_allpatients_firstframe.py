# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:26:45 2023

@author: Abdul Qayyum
"""


#%% test for long axis from dicom to nifiti conversion
##################### long axis
import os
import os
import re
import pickle
import cv2
import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import nibabel as nib

class BaseImage(object):
    """ Representation of an image by an array, an image-to-world affine matrix and a temporal spacing """
    volume = np.array([])
    affine = np.eye(4)
    dt = 1

    def WriteToNifti(self, filename):
        nim = nib.Nifti1Image(self.volume, self.affine)
        nim.header['pixdim'][4] = self.dt
        nim.header['sform_code'] = 1
        nib.save(nim, filename)
def find_series(dir_name, T):
    """
        In a few cases, there are two or three time sequences or series within each folder.
        We need to find which series to convert.
        """
    files = sorted(os.listdir(dir_name))
    if len(files) > T:
        # Sort the files according to their series UIDs
        series = {}
        for f in files:
            d = dicom.read_file(os.path.join(dir_name, f))
            suid = d.SeriesInstanceUID
            if suid in series:
                series[suid] += [f]
            else:
                series[suid] = [f]
        else:
            choose_suid = sorted(series.keys())[-1]
        print('There are multiple series. Use series {0}.'.format(choose_suid))
        files = sorted(series[choose_suid])

    if len(files) < T:
        print('Warning: {0}: Number of files < CardiacNumberOfImages! '
              'We will fill the missing files using duplicate slices.'.format(dir_name))
    return(files)


import os
#y=0
def LAX_dicom_nfti(path,name):
    #len_path=os.listdir(path)
    Z=1
    data={}
    #name='ch2'
    y=0
    z=path
    len_z=os.listdir(z)
    #print(len_z)
    d = dicom.read_file(os.path.join(z,len_z[0])) ### get first dicom from each folder
    #name=os.path.join(z,len_z[0]).split('\\')[-1]
    T = d.CardiacNumberOfImages
    # Read a dicom file from the correct series when there are multiple time sequences
    d = dicom.read_file(os.path.join(z, find_series(z, T)[0])) ##### take first file
    X = d.Columns
    Y = d.Rows
    T = d.CardiacNumberOfImages
    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])
       
    # The coordinate of the upper-left voxel of the first and second slices
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]
    axis_z = np.cross(axis_x, axis_y)

    # Determine the z spacing
    if hasattr(d, 'SliceThickness'):  #SliceThickness repalce SpacingBetweenSlices
        dz = float(d.SliceThickness)
    else:
        print('Warning: can not find attribute SpacingBetweenSlices. '
                  'Use attribute SliceThickness instead.')
        dz = float(d.SliceThickness)

    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3, 0] = axis_x * dx
    affine[:3, 1] = axis_y * dy
    affine[:3, 2] = axis_z * dz
    affine[:3, 3] = pos_ul
    #print(affine)
    # The 4D volume
    volume = np.zeros((X, Y, Z, T), dtype='float32')  # Height, width, space, time
    # Go through each slice
    # In a few cases, there are two or three time sequences or series within each folder.
    # We need to find which seires to convert.
    files = find_series(z, T)
    # Now for this series, sort the files according to the trigger time.
    files_time = []
    for f in files:
        d = dicom.read_file(os.path.join(z, f))
        t = d.TriggerTime
        files_time += [[f, t]]
    files_time = sorted(files_time, key=lambda x: x[1])
        
        
    # # Read the images thorugh time in our case it will be normally 25
    for t in range(0, T):
        #print(t)
        # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
        # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
        # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
        # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
        # with nibabel's dimension.
        try:
            f = files_time[t][0]
            d = dicom.read_file(os.path.join(z, f))
            volume[:, :, y, t] = d.pixel_array.transpose()
            #volume[:, :, z, t] = d.pixel_array.transpose()
            print(z)
        except IndexError:
            print('Warning: dicom file missing for {0}: time point {1}. '
                      'Image will be copied from the previous time point.'.format(z, t))
            volume[:, :, y, t] = volume[:, :, y, t - 1] ### y is contour for space z 
                #print(z)
            
        except (ValueError, TypeError):
            print('Warning: failed to read pixel_array from file {0}. '
                      'Image will be copied from the previous time point.'.format(os.path.join(z, f)))
            volume[:, :, y, t] = volume[:, :, y, t - 1]
        except NotImplementedError:
            print('Warning: failed to read pixel_array from file {0}. '
                      'pydicom cannot handle compressed dicom files. '
                      'Switch to SimpleITK instead.'.format(os.path.join(z, f)))
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(z, f))
            img = sitk.GetArrayFromImage(reader.Execute())
            volume[:, :, y, t] = np.transpose(img[0], (1, 0))

           
    # Temporal spacing between each time
    dt = (files_time[1][1] - files_time[0][1]) * 1e-3
    # Store the image
    volume=volume[:,:,:,0]   ##### for first frame1
    #print(volume.shape)
    data[name] = BaseImage()
    data[name].volume = volume
    data[name].affine = affine
    data[name].dt = dt
    y=y+1
    return data,affine,volume,dt

# path='C:\\Users\\aq22\\Desktop\kcl2022\\dataset_2023\\TestSAX\\patient_002_001_LAX\\CH2\\series0007-Body'
# name='test1'

def convert_dicom_to_nifti(output_dir,data):
    """ Save the image in nifti format. """
    for name, image in data.items():
        image.WriteToNifti(os.path.join(output_dir, '{0}'.format(name)))
import os
import glob
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
import numpy as np
import pandas as pd
import os
import pydicom
import cv2
from skimage import io
import natsort
# pathim='C:/Users/aq22/Desktop/kcl2022/dataset_2023/dataset_conversion_2023/LAX_datapath_1_new.csv'
# pthpn=pd.read_csv(pathim)
# #print(pthpn.columns)
# p_id=pthpn['patient_id']
# ch_2=pthpn['2ch_num']
pathdata='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\test2ch\\ch2'
output_dir='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\test2ch\\output'
patient=natsort.natsorted(os.listdir(pathdata))
#%
##################### mask file loading###################
import glob
for i in range(0,len(patient)):
    # #print(i)
    p_id=patient[i]
    #print(p_id)
    #ch_2=pthpn['2ch_num'][i] ################ 2CH #####
    path=os.path.join(pathdata,os.path.join(p_id))
    print(path)
    #break
    namep=path.split('\\')[-2]
    len_z=os.listdir(path)
    print(len_z[0])
    name=len_z[0].replace('dcm','nii.gz')
    fullname=namep+name
    fdir=os.path.join(output_dir,namep)
    createFolder(fdir)
    #print(len_z)
    data,affine,volume,dt=LAX_dicom_nfti(path,name)
    convert_dicom_to_nifti(fdir,data)
    #break
#%%   ch4 dicom to nifiti conversion it convert hxwx1xtime e.g 224x180x1x25
pathdata='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\test2ch\\ch4'
output_dir='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\test2ch\\output'
patient=natsort.natsorted(os.listdir(pathdata))
#%
##################### mask file loading###################
import glob
for i in range(0,len(patient)):
    # #print(i)
    p_id=patient[i]
    #print(p_id)
    #ch_2=pthpn['2ch_num'][i] ################ 2CH #####
    ##patint
        #series
        
    path=os.path.join(pathdata,os.path.join(p_id)) # this path represnt patient 
    #and then inside patient we have another folder is series
    print(path)
    #break
    namep=path.split('\\')[-2]
    len_z=os.listdir(path)
    print(len_z[0])
    name=len_z[0].replace('dcm','nii.gz')
    fullname=namep+name
    fdir=os.path.join(output_dir,namep)
    createFolder(fdir)
    #print(len_z)
    data,affine,volume,dt=LAX_dicom_nfti(path,name)
    convert_dicom_to_nifti(fdir,data)
    #break