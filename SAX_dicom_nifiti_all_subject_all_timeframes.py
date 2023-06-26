# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:39:30 2023

@author: aq22
"""

import os
import re
import pickle
import cv2
import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from glob import glob
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

def repl(m):
    """ Function for reformatting the date """
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))


def process_manifest(name, name2):
    """
        Read the lines in the manifest.csv file and check whether the date format contains
        a comma, which needs to be removed since it causes problems in parsing the file.
        """
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)


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

            if not find_series:
                    choose_suid = sorted(series.keys())[-1]
            else:
                choose_suid = sorted(series.keys())[-1]
            print('There are multiple series. Use series {0}.'.format(choose_suid))
            files = sorted(series[choose_suid])

        if len(files) < T:
            print('Warning: {0}: Number of files < CardiacNumberOfImages! '
                  'We will fill the missing files using duplicate slices.'.format(dir_name))
        return(files)
    

def dicom_nifti(path):
    # subdir = sorted(os.listdir(path))
    # #print(subdir)
    # ############# first slice and first dicom image
    # pathsfirst=os.path.join(path,subdir[0])
    # print(pathsfirst)
    # #print(glob(os.path.join(pathsfirst,'*.dcm')))
    # file_first=sorted(glob(os.path.join(pathsfirst,'*.dcm')))[0]
    # ############## second slice and first dicom image
    # pathsecond=os.path.join(path,subdir[1])
    # file_second=sorted(glob(os.path.join(pathsecond,'*.dcm')))[0]
    
    subdir = sorted(os.listdir(path))
    #print(subdir)
    pathsfirst=os.path.join(path,subdir[0])
    #print(pathsfirst)
    dicom_list_first = [ f for f in  os.listdir(pathsfirst)]
    #break
    #print(glob(os.path.join(pathsfirst,'*.dcm')))
    file_first=os.path.join(pathsfirst,dicom_list_first[0])
    ############## second slice and first dicom image
    pathsecond=os.path.join(path,subdir[1])
    dicom_list_second = [ f for f in  os.listdir(pathsecond)]
    file_second=os.path.join(pathsecond,dicom_list_second[0])

    data = {}
    name=path.split('\\')[-1]
    affinef=[]
    """ Read dicom images and store them in a 3D-t volume. """
    for file in sorted(subdir):
        #print(dir)
        dir=os.listdir(os.path.join(path,file))
        print(dir[0])
        #break
        #dirname=os.path.join(path,file)
        #break
        # Read the image volume
        # Number of slices
        Z = len(subdir)

        # Read a dicom file at the first slice to get the temporal information
        # We need the number of images in a sequence to check whether multiple sequences are recorded
        d = dicom.read_file(os.path.join(os.path.join(path,file),dir[0]))
        #d = dicom.read_file(os.path.join(dir[0], sorted(os.listdir(dir[0]))[0]))
        T = d.CardiacNumberOfImages
        #print(T)
        #print(int(T))
        #T1=int(T)
        #break

        # Read a dicom file from the correct series when there are multiple time sequences
        d = dicom.read_file(file_first)
        X = d.Columns
        Y = d.Rows
        T = d.CardiacNumberOfImages
        dx = float(d.PixelSpacing[1])
        dy = float(d.PixelSpacing[0])
        #break

        # DICOM coordinate (LPS)
        #  x: left
        #  y: posterior
        #  z: superior
        # Nifti coordinate (RAS)
        #  x: right
        #  y: anterior
        #  z: superior
        # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
        # Refer to
        # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
        # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

        # The coordinate of the upper-left voxel of the first and second slices
        pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
        pos_ul[:2] = -pos_ul[:2]

        # Image orientation
        axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
        axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
        axis_x[:2] = -axis_x[:2]
        axis_y[:2] = -axis_y[:2]

        if Z >= 2:
                # Read a dicom file at the second slice
                #d2 = dicom.read_file(os.path.join(dir[1], sorted(os.listdir(dir[1]))[0]))
                #d2 = dicom.read_file(os.path.join(os.path.join(pathsax,subdir[1]),dir[0]))
                d2 = dicom.read_file(file_second)
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
        else:
                    axis_z = np.cross(axis_x, axis_y)
            
    
        #axis_z = np.cross(axis_x, axis_y)

        # Determine the z spacing
        if hasattr(d, 'SpacingBetweenSlices'):
            dz = float(d.SpacingBetweenSlices)
        # elif Z >= 2:
            #     print('Warning: can not find attribute SpacingBetweenSlices. '
            #                  'Calculate from two successive slices.')
            #     dz = float(np.linalg.norm(pos_ul2 - pos_ul))
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
        affinef.append(affine)
        #break

        # The 4D volume
        volume = np.zeros((X, Y, Z, T), dtype='float32')
            
        # Go through each slice/folder
        for z in range(0, Z):
            # In a few cases, there are two or three time sequences or series within each folder.
            # We need to find which seires to convert.
            #files = find_series(dir[z], T)
            files = find_series(os.path.join(path,subdir[z]), T)
            #break
            # Now for this series, sort the files according to the trigger time.
            files_time = []
            for f in files:
                d = dicom.read_file(os.path.join(os.path.join(path,subdir[z]), f))
                t = d.TriggerTime
                files_time += [[f, t]]
                files_time = sorted(files_time, key=lambda x: x[1])

            # Read the images
            for t in range(0, T):
                # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
                # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
                # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
                # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
                # with nibabel's dimension.
                try:
                    f = files_time[t][0]
                    d = dicom.read_file(os.path.join(os.path.join(path,subdir[z]), f))
                    volume[:, :, z, t] = d.pixel_array.transpose()
                except IndexError:
                    print('Warning: dicom file missing for {0}: time point {1}. '
                              'Image will be copied from the previous time point.'.format(dir[z], t))
                    volume[:, :, z, t] = volume[:, :, z, t - 1]
                except (ValueError, TypeError):
                    print('Warning: failed to read pixel_array from file {0}. '
                              'Image will be copied from the previous time point.'.format(os.path.join(path,subdir[z]), f))
                    volume[:, :, z, t] = volume[:, :, z, t - 1]
                except NotImplementedError:
                    print('Warning: failed to read pixel_array from file {0}. '
                              'pydicom cannot handle compressed dicom files. '
                              'Switch to SimpleITK instead.'.format(os.path.join(path,subdir[z]), f))
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(os.path.join(os.path.join(path,subdir[z]), f))
                    img = sitk.GetArrayFromImage(reader.Execute())
                    volume[:, :, z, t] = np.transpose(img[0], (1, 0))
                
                
        # Temporal spacing
        dt = (files_time[1][1] - files_time[0][1]) * 1e-3

        # Store the image
        data[name] = BaseImage()
        data[name].volume = volume
        data[name].affine = affine
        data[name].dt = dt
        
    return data,volume,affine,dt,affinef

#path='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\TestSAX\\patient_002_001_SAX'

############### loop over multiple subject to convert dicom to nifiti

# pathim='C:/Users/aq22/Desktop/kcl2022/dataset_2023/dataset_conversion_2023/SAX_datapath_new.csv'
# pthpn=pd.read_csv(pathim)
# #print(pthpn.columns)
# p_id=pthpn['patient_id']
# sax=pthpn['sax_num']
pathdata='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\testSAX'
lstdata=os.listdir(pathdata)
#%
####SAX_rawdata
       #subject1
            #seriesfolder1
                       ### dicom1,.........dicom25
            #seriesfolder2
                       ### dicom1,.........dicom25
                       
           #seriesfolder15
                      ### dicom1,.........dicom25
                      
      #subject2
           #seriesfolder1
                      ### dicom1,.........dicom25
           #seriesfolder2
                      ### dicom1,.........dicom25
                      
          #seriesfolder15
                     ### dicom1,.........dicom25
          # .
          # .
           #.
     #subject20
          #seriesfolder1
                     ### dicom1,.........dicom25
          #seriesfolder2
                     ### dicom1,.........dicom25
                     
         #seriesfolder15
                    ### dicom1,.........dicom25

##################### mask file loading###################
import glob
# pathGT='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\GT_new_files\\CVIfiles_combined'
# pathgtfile=glob.glob(os.path.join(pathGT,'*'))
def convert_dicom_to_nifti(output_dir,data):
    """ Save the image in nifti format. """
    for name, image in data.items():
        image.WriteToNifti(os.path.join(output_dir, '{0}.nii.gz'.format(name)))

output_dir='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\dataset_conversion_2023\\test2ch\\output\\SAX\\'
#%
for i in range(0,len(lstdata)):
    path=os.path.join(pathdata,lstdata[i])
    #print(path)
    name=path.split('\\')[-1]
    print(name)
    #break
    # # print(path)
    # subdir = sorted(os.listdir(path))
    # #print(subdir)
    # pathsfirst=os.path.join(path,subdir[0])
    # #print(pathsfirst)
    # dicom_list_first = [ f for f in  os.listdir(pathsfirst)]
    # #break
    # #print(glob(os.path.join(pathsfirst,'*.dcm')))
    # file_first=os.path.join(pathsfirst,dicom_list_first[0])
    # ############## second slice and first dicom image
    # pathsecond=os.path.join(path,subdir[1])
    # dicom_list_second = [ f for f in  os.listdir(pathsecond)]
    # file_second=os.path.join(pathsecond,dicom_list_second[0])
    
    #break
    data,volume,affine,dt,affinef=dicom_nifti(path)
    #break
    #vvfs1=volume[:,:,:,0]  ############# take time=0,comment it for other time point
    data[name] = BaseImage()
    data[name].volume = volume
    data[name].affine = affine
    data[name].dt = dt
    convert_dicom_to_nifti(output_dir,data)
    #break
# #%%
# name=path.split('\\')[-1]
# data,volume,affine,dt=dicom_nifti(path)
# vvfs1=volume[:,:,:,0]
# #%
# #vvfs1s11=np.swapaxes(vvfs1,1,0)
# ########################### convert data into nifiti
# data[name] = BaseImage()
# data[name].volume = vvfs1
# data[name].affine = affine
# data[name].dt = dt
# def convert_dicom_to_nifti(output_dir,data):
#     """ Save the image in nifti format. """
#     for name, image in data.items():
#         image.WriteToNifti(os.path.join(output_dir, '{0}_new2.nii.gz'.format(name)))
# output_dir='C:\\Users\\aq22\\Desktop\\kcl2022\\dataset_2023\\TestSAX\\nifftt'
# convert_dicom_to_nifti(output_dir,data)