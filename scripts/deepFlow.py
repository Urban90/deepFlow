#!/usr/bin/env python3
# Import libraries
import cv2
import easygui as eg
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import multiprocessing as mp
import multiprocessing as mp
import nibabel as nib
import nibabel.nicom.csareader as csa
import nibabel.nicom.dicomreaders as readers
import nibabel.nicom.dicomwrappers as wrapper
import numpy as np
import numpy as np
import numpy as np
import os
import os
import pandas as pd
import pydicom
import random
import re
import scipy as sp
import scipy.ndimage as nd
import scipy.stats
import shutil
import string
import tensorflow as tf
import tqdm
import zipfile
from ast import literal_eval
from itertools import chain
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.merge import concatenate, add
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import curve_fit
from skimage.io import imread, imshow, concatenate_images
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm_notebook, tnrange
import warnings
warnings.filterwarnings('ignore')
# Define software identity
software = "DeepFlow"
version = "0.0.1alpha"

# Definitions
def extract_data(input_path, extracted_path, splitted):
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extracted_path, splitted))
    return
def anon(extracted_path):
    count = 0
    for filename in os.listdir(extracted_path):
        x = os.path.join(extracted_path, filename)
        for subfile in os.listdir(x):
            y = os.path.join(x, subfile)
            if subfile.endswith(".dcm"):
                count += 1
                ds = pydicom.dcmread(y, force = True)
                new_name = filename    #.split('_')[-4][-7:]
                try:
                    ds[0x0010, 0x0010].value = new_name  # Patient's Name
                    ds[0x0010, 0x0020].value = new_name  # Patient's Name
                    pydicom.dcmwrite(y, ds)
                except:
                    print(count, 'ERROR:', filename)
                    pass                  
def rename_filename(extracted_path):
    for r, d, f in os.walk(extracted_path):
        for i, file in enumerate(f): 
            try:
                x = os.path.join(r, file)
                ds = pydicom.dcmread(x, force= True)
                new_name = ds[0x0010, 0x0010].value
                instance = ds[0x0020, 0x0013].value
                description = ds[0x0008, 0x103E].value
                description = description.split('@')[-1]
                #print(new_name)
                #print(instance)
                #print(description)
                #print('trying to rename the filename :', new_name)
                os.rename(x, os.path.join(r, str(new_name)+'_'+str(description)+'_'+str(instance)+'.dcm'))
            except:
                print('error: ', x)
#there is so manifest in extracted_path
def delete_manifest(extracted_path):
    for filename in os.listdir(extracted_path):
        x = os.path.join(extracted_path, filename)
        for subfile in os.listdir(x):
            y = os.path.join(x, subfile)
            #if 'manifest' in subfile:
                #print(subfile)
            if 'manifest' in subfile:
                os.system(f'rm -r {y}')
                #print('removing manifest files :', subfile)
                #print(y)
def create_matrix(extracted_path):
    matrix_dicoms = []
    #for r, d, f in os.walk(extracted_path):
    for file in os.listdir(extracted_path):
        x = os.path.join(extracted_path, file)
        for filename in os.listdir(x):
            y = os.path.join(x, filename)
            try:
                ds = pydicom.dcmread(y, force= True)
            #new_name = r.split('_')[-4][-7:]
            #instance = ds[0x0020, 0x0013].value
            #description = ds[0x0008, 0x103E].value
            #description = description.split('@')[-1]
                print('creating dicom matrices :', file)
                matrix_dicoms.append(ds)
            except:
                print('following dicom matrix did not work: ', x)
        return matrix_dicoms
def create_submatrix(matrix_dicoms):
    matrices_ds_c = []
    matrices_arrays = []
    for ds in matrix_dicoms:
        description = ds[0x0008, 0x103E].value
        description = description.split('@')[-1]
        if description == 'c':
            matrices_arrays.append(ds.pixel_array)
        if description == 'c':
            matrices_ds_c.append(ds)
    matrices_ds_c = np.array(matrices_ds_c)
    matrices_arrays = np.array(matrices_arrays)
    return matrices_ds_c, matrices_arrays
def brightness_contrast(img, brightness, contrast):
    slope=math.tan((math.pi * (contrast/100.0+1.0)/4.0))
    if slope < 0.0:
        slope=0.0
    intercept=brightness/100.0+((100-brightness)/200.0)*(1.0-slope)
    img = img*slope + intercept
    return img
def aortaHunt_dicom(h5, ds, masks_path_2):
    model = tf.keras.models.load_model(h5)
    #os.system(f"magick convert -brightness-contrast 5x25 {image} {image[:-4]}_conv.tif")
    X = np.zeros((1, 128, 128, 1), dtype=np.float32)
    #pic = load_img(image[:-4] + "_conv.tif", color_mode = "grayscale")
    #pic = load_img(image, color_mode = "grayscale")
    #os.system(f"rm  {image} {id_[:-4]}.tif")
    #x_img = MyImage()
    x_img = ds.pixel_array
    x_img = brightness_contrast(x_img, 5, 25)
    x_img = img_to_array(x_img)
    x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    X[0] = x_img / 255.0
    new_name = ds[0x0010, 0x0010].value
    instance = ds[0x0020, 0x0013].value
    description = ds[0x0008, 0x103E].value
    description = description.split('@')[-1]
    #plt.imshow(np.squeeze(x_img), cmap = plt.cm.bone)
    #plt.title(str(new_name)+'_'+str(description)+'_'+str(instance))
    #plt.show()
    result = model.predict(X)
    #plt.imshow(np.squeeze(result), cmap = plt.cm.bone, alpha = 0.4)
    #plt.show()
    #result = (result > 0.5).astype(np.uint8)
    mask = np.squeeze(result)
    x = os.path.join(masks_path_2, str(new_name)+'_'+str(description)+'_'+str(instance)+'.jpg')
    plt.imsave(x, mask, format = 'jpg', cmap = plt.cm.bone)
    return mask
def get_masks(matrices_ds_c, model_aorta, masks_path_2):
    masks = []
    for img in matrices_ds_c:
        #print(i)
        mask = aortaHunt_dicom(model_aorta, img, masks_path_2)
        masks.append(mask)
    return masks
def get_noise(rotated):
    img0 = rotated[0]
    img1 = rotated[1]
    img2 = rotated[2]
    img3 = rotated[3]
    img4 = rotated[4]
    img5 = rotated[5]
    img6 = rotated[6]
    img7 = rotated[7]
    img8 = rotated[8]
    img9 = rotated[9]
    img10 = rotated[10]
    img11 = rotated[11]
    img12 = rotated[12]
    img13 = rotated[13]
    img14 = rotated[14]
    img15 = rotated[15]
    img16 = rotated[16]
    img17 = rotated[17]
    img18 = rotated[18]
    img19 = rotated[19]
    img20 = rotated[20]
    img21 = rotated[21]
    img22 = rotated[22]
    img23 = rotated[23]
    img24 = rotated[24]
    img25 = rotated[25]
    img26 = rotated[26]
    img27 = rotated[27]
    img28 = rotated[28]
    img29 = rotated[29]
    final = np.zeros((192,192), dtype=np.float32)
    #now, let try to make pixel-wise operations
    h = final.shape[0]
    w = final.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            value0 = img0[y, x] 
            value1 = img1[y, x] 
            value2 = img2[y, x] 
            value3 = img3[y, x] 
            value4 = img4[y, x] 
            value5 = img5[y, x] 
            value6 = img6[y, x] 
            value7 = img7[y, x]
            value8 = img8[y, x] 
            value9 = img9[y, x] 
            value10 = img10[y, x] 
            value11 = img11[y, x] 
            value12 = img12[y, x] 
            value13 = img13[y, x] 
            value14 = img14[y, x] 
            value15 = img15[y, x] 
            value16 = img16[y, x] 
            value17 = img17[y, x] 
            value18 = img18[y, x]
            value19 = img19[y, x] 
            value20 = img20[y, x] 
            value21 = img21[y, x]
            value22 = img22[y, x] 
            value23 = img23[y, x] 
            value24 = img24[y, x] 
            value25 = img25[y, x]
            value26 = img26[y, x] 
            value27 = img27[y, x]
            value28 = img28[y, x] 
            value29 = img29[y, x] 
            list_values = [value0,value1,value2,value3,value4,value5,value6,value7,value8,value9,value10,value11,value12,value13,value14,value15,value16,value17,value18,value19,value20,value21,value22,value23,value24,value25,value26,value27,value28,value29]
            stdv = np.min(list_values)
            #stdv = abs(stdv)
            final[y,x] = stdv
    #final_cropped = final[10:182, 10:182]
    flatten = list(final.flatten())
    value = stats.percentileofscore(flatten, float(35))
    #print('this is the min stdv:', np.min(final))
    b = np.floor(np.percentile(flatten, 0.05))
    #print(value)
    #print(b)
    final = np.array(final)
    thresh = cv2.threshold(final, b, 1, cv2.THRESH_BINARY)[1]
    
    
    #thresh = np.pad(thresh, 10, mode='constant')
    #print('shape:', thresh.shape)
    #masked = thresh*rotated
    thresh = thresh.astype(np.int64)
    rotated = rotated.astype(np.int64)
    #plt.imshow(thresh, cmap = 'jet_r')
    #plt.title('thresh')
    #plt.show()
    ano = []
    for i in rotated:
        img = i*thresh
        #plt.imshow(img)
        #plt.show()
        ano.append(img)
    ano = np.array(ano)
    #ano = ano/(250/4096)
    #plt.imshow(masked[0])
    #plt.show()
    
    return ano
def get_param(file):
    dcm = file
    #dcm = dicom.dcmread(file, force = True)
    num = str(dcm[0x0010, 0x0020].value)+'_'+str(dcm[0x0020, 0x0013].value)
    pix_dim = dcm[0x0028, 0x0030].value
    pix_dim = [float(a) for a  in pix_dim]
    matrix = max(dcm[0x0018, 0x1310].value)
    interval = float(dcm[0x0018,0x1062].value)
    c = dcm[0x0029, 0x1010].value
    brand = dcm[0x0008, 0x0070].value
    rescale_inter = dcm[0x0028, 0x1052].value
    rescale_slope = dcm[0x0028, 0x1053].value
    if dcm[0x0008, 0x0070].value == 'SIEMENS':
        try:
            venc = csa.read(c)['tags']['FlowVenc']['items'][0]
            last = 1
            #print('venc can be retrieved as is set to: ', venc)
        except:
            venc = 250
            last = 0
            #print('venc can not be retrieved  - it is set to 250')
    else:
        print('error', brand)
    bits = dcm[0x0028, 0x0101].value
    params = [num, brand, matrix, pix_dim[0], pix_dim[1], venc, interval, bits, last, rescale_inter, rescale_slope]
    return params
def create_params(matrix_dicoms, out_file):
    list_params = []
    for i in matrix_dicoms:
        if i[0x0008, 0x103E].value == 'flow_250_tp_AoV_bh_ePAT@c_P':
            params = get_param(i)
            list_params.append(params)
    out = pd.DataFrame(list_params, columns=['ind', 'brand', 'matrix', 'dim1', 'dim2','venc', 'interval', 'p1', 'p2', 'rescale_int', 'rescale_slope'])
    out.to_csv(out_file, index=False)
def move_files_nifti(path, masks_path_2, nifti_final):
    filenames = []
    for filename in os.listdir(masks_path_2):
        x = filename.split('_')[0]
        filenames.append(x)
    filenames = set(filenames)
    for i in filenames:
        os.mkdir(os.path.join(nifti_final, i))
    for filename in os.listdir(masks_path_2):
        x = filename.split('_')[0]
        shutil.copy(os.path.join(masks_path_2, filename), os.path.join(os.path.join(nifti_final, x), filename))
    for i in os.listdir(path):
        x = os.path.join(path, i)
        patient = i.split('_')[0]
        for j in os.listdir(x):
            y = os.path.join(x, j)
            #print(y)
            
            #print(j)
            ds = pydicom.dcmread(y, force=True)
            #if ds[0x0008, 0x103E].value == 'flow_250_tp_AoV_bh_ePAT@c_P':
            if '_c_P_' in j:
                shutil.copy(y, os.path.join(os.path.join(nifti_final, patient), j))
def get_bb(img, out_file):
    p = pd.read_csv(out_file)
    dt = p['interval'].values[0]/30/1000 # how much time in seconds in one frame
    pix_dim = p['dim1'].values[0]/10
# get contours
    result = img.copy()
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (30, 100, 255), 2)
        #print("x,y,w,h:",x,y,w,h)
    #print(contours)
    area = np.sum(img)*pix_dim**2
    #print('this is the area :', area)
    
    #plt.imshow(result, cmap = plt.cm.bone)
    #plt.show()
    width = w*pix_dim
    height = h*pix_dim
    #print('this is the diameter :', width)
    if width > height:
        perimeter = math.pi*((3*(width/2+height/2))-math.sqrt((3*width/2+height/2)*(width/2+3*height/2)))
        #print('this is the perimeter :', perimeter)
        return perimeter, width, height, area
    else:
        
        perimeter = math.pi*((3*(height/2+width/2))-math.sqrt((3*height/2+width/2)*(height/2+3*width/2)))
        #print('this is the perimeter :', perimeter)
        return perimeter, height, width, area
#definition to create nifti file
def dicom2nii(path_folder):
    os.system(f"dcm2niix -f %i {path_folder}")
def get_flow(masked, out_file):
    p = pd.read_csv(out_file)
    dt = p['interval'].values[0]/30/1000 # how much time in seconds in one frame
    pix_dim = p['dim1'].values[0]/10 #in cms
    
    masked = masked/dt
    np_sum = []
    for i,j in enumerate(masked):
        soma = np.sum(j)
        np_sum.append(soma)
    X = list(range(30))
    X = [(x*dt + dt)*1000 for x in X]
    #plt.plot(X, np_sum)
    #plt.axhline(y=0, color='r', linestyle='-')
    #plt.title('ml/s flow')
    #plt.show()
    
    masked = masked * dt
    flow_sum = []
    antegrade_sum = []
    retrograde_sum = []
    for frame in masked:
        flow = np.sum(frame)
        flow_sum.append(flow)
        if flow >= 0:
            antegrade_sum.append(flow)
        else:
            retrograde_sum.append(flow)
    
    #plt.plot(flow_sum)
    #plt.axhline(y=0, color='r', linestyle='-')
    #plt.title('ml flow')
    #plt.show()   
    #print('the forward - retrograd volume is: ', np.sum(flow_sum))
    #print('the regurgitant volume is: ', np.sum(retrograde_sum))
    #print('the stroke volume is: ', np.sum(antegrade_sum))
    #print('the regurgitation fraction is :', (abs(np.sum(retrograde_sum))/np.sum(antegrade_sum))*100)
    net_flow = np.sum(flow_sum)
    retrograde_flow = np.sum(retrograde_sum)
    antegrade_flow = np.sum(antegrade_sum)
    regurgitation_fraction = ((abs(np.sum(retrograde_sum)))/np.sum(antegrade_sum))*100
    boolean = antegrade_flow > abs(retrograde_flow)
    #print(boolean)
    
    
    return net_flow, retrograde_flow, antegrade_flow, regurgitation_fraction, boolean
def asf(img, out_file):
    p = pd.read_csv(out_file)
    dt = p['interval'].values[0]/30/1000 # how much time in seconds in one frame
    pix_dim = p['dim1'].values[0]/10
    venc = p['venc'].values[0]
    a = ((pix_dim))/dt
    final_2 = np.zeros((192,192), dtype=np.float32)
    for y in range(0, 192):
        for x in range(0, 192):
            #if img[y,x] == 0:
                #value = img[y,x]
            #else:
            value = img[y,x]*(venc/4096) #*a
            #value = value - 4096
            #value = value*0.06
            #value = (img[y,x]*2) 
            #value = value - 4096
            #value = value/4096
            #value = value*4096
            #value = value/(2*0.087)
            final_2[y,x] = value
    return final_2
 
def loadnii(nii_path, out_file):
    p = pd.read_csv(out_file)
    dt = p['interval'].values[0]/30/1000 # how much time in seconds in one frame
    pix_dim = p['dim1'].values[0]/10
    nifti_files_list = []
    nifti_ids = []
    masks = []
    perimeters = []
    max_diameters = []
    min_diameters = []
    areas = []
    for i in os.listdir(nii_path):
        x = os.path.join(nii_path, i)
        if i.endswith('.nii'):
            #nifti_files = nib.as_closest_canonical(nib.load(x))
            nifti_files = nib.load(x)
            nifti_files = nifti_files.get_data().astype(np.int64)
            nifti_files_list.append(nifti_files)
            nifti_ids.append(i.split('_ph.nii')[0])
            print(i)
            #print(i)
        elif i.endswith('.jpg'):
            img = cv2.imread(x, 0)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]
            img = cv2.resize(img, (192,192), interpolation = cv2.INTER_AREA)
            p, max_dia, min_dia, area = get_bb(img, out_file)
            perimeters.append(p)
            max_diameters.append(max_dia)
            min_diameters.append(min_dia)
            areas.append(area)
        #print(img)
            masks.append(img)
    #nifti_files_list = np.array(np.squeeze(nifti_files_list))
    nifti_files_list = np.moveaxis(nifti_files_list, -1, 0)
    nifti_files_list = np.array(np.squeeze(nifti_files_list))
    aorta_perimeter = np.mean(perimeters)
    max_aorta = np.max(max_diameters)
    min_aorta = np.max(min_diameters)
    aorta_area = np.max(areas)
    #rotated = []
    #for i in nifti_files_list:
        #phase = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #phase = cv2.flip(i, 1)
        #rotated.append(phase)
    #rotated = np.array(rotated)
    neg_files = -nifti_files_list
    #z_values = get_stv(nifti_files_list)
    #z_rotated = []
    #for i in z_values:
        #z= cv2.rotate(i, cv2.ROTATE_180)  
        #z_rotated.append(z)
    #z_values = np.array(z_rotated)
    #neg_files = -nifti_files_list
    #flipped = []
    #for i in nifti_files_list:
        #phase = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #phase = cv2.flip(i, 1)
        #flipped.append(phase)
    #flipped = np.array(flipped)
    #nifti_files_list = cv2.rotate(nifti_files_list, cv2.ROTATE_90_COUNTERCLOCKWISE)
    nifti_ids = np.array(nifti_ids)
    
    #print('frame intervall is: ', dt)
    #print('pix_dim is: ', pix_dim)
    #print('this is venc: ', p['venc'].values[0])
    #print(nifti_files_list.shape)
    masks = np.array(masks)
    asfs = []
    for i in nifti_files_list:
        img = asf(i, out_file)
        asfs.append(img)
    nifti_files_list = np.array(asfs)

    nifti_files_list = np.array(nifti_files_list)
    
    neg_files = -nifti_files_list
    masked = nifti_files_list * masks  #*(pix_dim**2) *dt
    masked = get_noise(masked) #correction is here
    #plt.imshow(masked[7])
    #plt.title('after noise correction')
    #plt.show()
    zipped_masks = list(zip(masked, masks))
    sums_v = []
    for i,j in zipped_masks:
    #j = asf(j)
        area = np.sum(j)
    #print(np.max(j))
        mean_velocity = np.sum(i)
        v = mean_velocity/area
        sums_v.append(v)
    masks_v_min = np.min(sums_v)
    masks_v_max = np.max(sums_v)
    masked = masked *(pix_dim**2) *dt
    masked_neg = neg_files*masks  #*(pix_dim**2) *dt
    masked_neg = get_noise(masked_neg)
    masked_neg = masked_neg*(pix_dim**2) *dt
    
    zipped_masks_neg = list(zip(masked_neg, masks))
    sums_neg = []
    for i,j in zipped_masks_neg:
    #j = asf(j)
        area = np.sum(j)
        mean_velocity = np.sum(i)
        v = mean_velocity/area
        sums_neg.append(v)
    masks_neg_min = np.min(sums_neg)
    masks_neg_max = np.max(sums_neg)
    net_flow, retrograde_flow, antegrade_flow, regurgitation_fraction, boolean = get_flow(masked, out_file)
    net_flow_neg, retrograde_flow_neg, antegrade_flow_neg, regurgitation_fraction_neg, boolean_neg = get_flow(masked_neg, out_file)
    
    if boolean == True:
        #print('boolean is true and the stroke volume is: ',antegrade_flow )
        return nifti_ids, net_flow, retrograde_flow, antegrade_flow, regurgitation_fraction, aorta_perimeter, max_aorta, min_aorta, aorta_area, masks_v_max, masks_v_min

    else:
        #print('boolean is false and the stroke volume is: ',antegrade_flow_neg )
        return nifti_ids, net_flow_neg, retrograde_flow_neg, antegrade_flow_neg, regurgitation_fraction_neg,aorta_perimeter, max_aorta, min_aorta, aorta_area, masks_neg_max, masks_neg_min
    
def create_nifti(nifti_final):
    for i in os.listdir(nifti_final):
        dicom2nii(os.path.join(nifti_final, i))
def get_ai(nifti_final, out_file):
    ids = []
    regurgitation = []
    antegrade_flow_list =[]
    retrograde_flow_list = []
    net_flow_list = []
    aorta_perimeter_list = []
    max_aorta_list = []
    min_aorta_list = []
    aorta_area_list = []
    masks_v_max_list = []
    masks_v_min_list = []
    for i in os.listdir(nifti_final):
        path = os.path.join(nifti_final, i)
        nifti_ids, net_flow, retrograde_flow, antegrade_flow, regurgitation_fraction, aorta_perimeter, max_aorta, min_aorta, aorta_area, masks_v_max, masks_v_min = loadnii(path, out_file)
        ids.append(nifti_ids)
        regurgitation.append(regurgitation_fraction)
        net_flow_list.append(net_flow)
        antegrade_flow_list.append(antegrade_flow)
        retrograde_flow_list.append(retrograde_flow)
        aorta_perimeter_list.append(aorta_perimeter)
        max_aorta_list.append(max_aorta)
        min_aorta_list.append(min_aorta)
        aorta_area_list.append(aorta_area)
        masks_v_max_list.append(masks_v_max)
        masks_v_min_list.append(masks_v_min)
    return ids, regurgitation, antegrade_flow_list, retrograde_flow_list, net_flow_list, aorta_perimeter_list, max_aorta_list, min_aorta_list, aorta_area_list, masks_v_max_list, masks_v_min_list
def get_csv(ids, regurgitation, antegrade_flow_list, retrograde_flow_list, net_flow_list, aorta_perimeter_list, max_aorta_list, min_aorta_list, aorta_area_list, masks_v_max_list, masks_v_min_list, out_aorta):
    patient_id = str(ids[0])
    patient_id = patient_id.strip('[')
    patient_id = patient_id.strip(']')
    patient_id = patient_id.strip("''")
    patient_id = str(patient_id)
    #print(patient_id)
    aortic_regurgitation = pd.DataFrame()
    aortic_regurgitation['id'] = ids[0]
    aortic_regurgitation['regurgitation_fraction'] = regurgitation
    #aortic_regurgitation['PPG'] = means
    aortic_regurgitation['stroke_volume'] = antegrade_flow_list
    aortic_regurgitation['regurgitant_volume'] = retrograde_flow_list
    aortic_regurgitation['net_flow'] = net_flow_list
    aortic_regurgitation['aorta_perimeter'] = aorta_perimeter_list
    aortic_regurgitation['max_aorta'] = max_aorta_list
    aortic_regurgitation['min_aorta'] = min_aorta_list
    aortic_regurgitation['aorta_area'] = aorta_area_list
    aortic_regurgitation['max_velocity'] = masks_v_max_list
    aortic_regurgitation['min_velocity'] = masks_v_min_list
    aortic_regurgitation.to_csv(os.path.join(out_aorta, patient_id+'.csv'), index=False)
    return aortic_regurgitation
def pipeline(path, extracted_path, splitted, model_aorta, masks_path_2, nifti_final, out_file, out_aorta):
    extract_data(path, extracted_path, splitted)
    print('Files extracted')
    anon(extracted_path)
    print('Anon files successful')
    rename_filename(extracted_path)
    print('renaming successful')
    delete_manifest(extracted_path)
    print('deleting manifest file successful')
    matrix_dicoms = create_matrix(extracted_path)
    print('matrix_dicoms created')
    matrices_ds_c, matrices_arrays = create_submatrix(matrix_dicoms)
    print('submatrix created')
    masks = get_masks(matrices_ds_c, model_aorta, masks_path_2)
    print('masks successfully created')
    create_params(matrix_dicoms, out_file)
    print('parameters created')
    move_files_nifti(extracted_path, masks_path_2, nifti_final)
    print('files in nifti final successfully moved')
    create_nifti(nifti_final)
    print('nifti files sucessfully created')
    ids, regurgitation, antegrade_flow_list, retrograde_flow_list, net_flow_list, aorta_perimeter_list, max_aorta_list, min_aorta_list, aorta_area_list, masks_v_max_list, masks_v_min_list = get_ai(nifti_final, out_file)
    print('ar parameters retrieved')
    aortic_regurgitation = get_csv(ids, regurgitation, antegrade_flow_list, retrograde_flow_list, net_flow_list, aorta_perimeter_list, max_aorta_list, min_aorta_list, aorta_area_list, masks_v_max_list, masks_v_min_list, out_aorta)
    print('ar parameters exported')
    os.system(f'rm -r {extracted_path}/*')
    os.system(f'rm -r {nifti_final}/*')
    os.system(f'rm -r {masks_path_2}/*')
    os.system(f'rm -r {out_file}')
    print('Folders succesfully erased')
def main():
    #os.chdir('/media/bruna/Bruna/20213_january')
    eg.msgbox(f"Welcome to {software}!\nSelect RUN to choose the folder containing DICOM images!", title = f"{software}: {version}", ok_button= "Run")
    folder_path = eg.diropenbox("DICOMs", title = "Select Folder")
    if folder_path == None:
        eg.msgbox(f"You didn't provide a valid folder.\nTry again.\nThank you for using {software}!", title = f"{software}: Folder error", ok_button= "Exit")
        exit()
    else:
        pass
    #extracted_path_example = './extracted_path_example'
    extracted_path = '/output/extracted_path_final'
    mag = '/output/MAG/'
    jpg_path = '/output/jpg/'
    out_aorta= '/output/results/'
    out_file = '/output/params.csv'
    model_aorta = '/assets/model.h5'
    masks_path_2 = '/output/masks2/'
    nifti_final = '/output/nifti_final/'
    didnmake_folder = '/output/didnmake/'
    didnmake = f"{didnmake_folder}/List.txt"
    filenames_done = []
    os.system(f"mkdir -p {extracted_path} {mag} {jpg_path} {out_aorta} {masks_path_2} {nifti_final} {didnmake_folder}")
    for i in os.listdir(out_aorta):
        #filenames_done.append(i)
        i = i.split('.')[0]
        filenames_done.append(i)
        #print(i)
    with open(didnmake, "a") as failFile:
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            splitted = filename.split('.zip')[0]
            print(f"Working on: {splitted}")
            if splitted not in filenames_done:
                pipeline(path = path,
                extracted_path = extracted_path,
                splitted = splitted,
                model_aorta = model_aorta,
                masks_path_2 = masks_path_2,
                nifti_final = nifti_final,
                out_file = out_file,
                out_aorta = out_aorta)
    os.system(f"bash /concat.sh {out_aorta}")
    os.system(f"mv {out_aorta}/*.tsv /output/")
    os.system(f"rm -rf {mag} {jpg_path} {out_file} {masks_path_2} {nifti_final} {didnmake_folder} {extracted_path} {out_aorta}")
if __name__ == "__main__":
    main()