# test script for integrating Pilatus6M images at CHESS beamline ID4B
# written by Nathan S Johnson 05/16/2023

import pyFAI
import numpy as np
import cbf
import pandas as pd
import sys
import glob


# function for opening the diffraction images
# returns the image as a numpy array
def open_image_as_array(imgPath,dtype = 'uint32'):
    # open the image using the PIL library
    img = Image.open(imgPath)

    # convert to array
    imgarr = np.array(img,dtype=dtype)

    # return the array
    return imgarr

# function for integrating a 2D array of diffraction intensities into a 1D histogram
def data_reduction(imgPath, poniPath, exportPath, thbin = 10000, return_tth = False, label = None):

    # open the image as an array
    img = cbf.read(imgPath)
    imArray = img.data
    # set data type as unsigned 32 bit integer
    imArray = np.array(imArray,dtype='uint32')

    # open the poni file
    p = pyFAI.load(poniPath)

    # create a mask
    # get the shape of the image
    s1 = int(imArray.shape[0])
    s2 = int(imArray.shape[1])
    
    # create a mask -- one that masks out negative values
    # and one that masks out saturated pixels
    high_mask = np.ones((s1, s2)) * (imArray > 9.9995*10**5) # saturated pixels
    low_mask = np.ones((s1, s2)) * (imArray <= 0) # dead pixels
    detector_mask = low_mask + high_mask
  
    # define the unit to return (two-theta or q) 
    if return_tth == True:
        unit = '2th_deg'
    else:
        unit = 'q_A^-1'

    # perform the integration
    Qlist, IntAve = p.integrate1d(imArray, thbin, mask = detector_mask,unit=unit)

    # now export the integrated 1d histogram
    df = pd.DataFrame({'x':Qlist, 'I':IntAve})
    df.to_csv(exportPath + '/' +  label + '_integrated.csv',index=False)

    return IntAve

def integrate_images(folderPath, poniPath, exportPath):
    # get the current directory
    currDir = os.getcwd()

    # switch to raw data directory
    os.chdir(folderPath)

    # glob all of the files
    imgPaths = glob.glob('*.cbf')

    all_res = []

    # go through each image and integrate
    for ii,imgPath in enumerate(imgPaths):
	# create a label
        label = imgPath[0:-4] + '_integrated_' + str(ii)
        print(label)

        res = data_reduction(imgPath, poniPath, exportPath + label + '.csv')
        all_res.append(res)

    # switch back to the original directory
    os.chdir(currDir) 

    # return the list of intensity arrays
    return all_res

if __name__ == "__main__":
    args = sys.argv
    integrate_images(args[1], 'lab6_17keV.poni', args[2])
	
