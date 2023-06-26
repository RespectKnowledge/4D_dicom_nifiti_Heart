# 4D_dicom_nifiti_Heart

#This code will convert dicom to nifiti for 3D and 4D Short-axis and long axis views

For 3D case we have output for SAX (HxWXD) e,g: 224x224x13 and for LAX(2CH,3CH,4CH) HxWxD(224x224x1)

If we consider time frame as well.For SAX output will be HxWxDXT(224x224x12x25) and LAX(2CH,3CH,4CH) HxWxDxT(224x224x1x25)


# For 4D LAX run this

LAX_2ch_4ch_dicom_nifiti_allpatients_alltimeframes.py

# For 4D SAX run this

SAX_dicom_nifiti_all_subject_all_timeframes.py


# For 3D (only for first time frame) LAX run this

LAX_2ch_4ch_dicom_nifiti_allpatients_firstframe.py

# For 3D (only for first time frame) SAX run this

SAX_dicom_nifiti_all_subject_all_firstframe.py

# Data sample is provided for one patient as an example. You can set folder or directory according to your dataset.

Any question, please email me
engr.qayyum@gmail.com
a.qayyum@imperial.ic.ac.uk





