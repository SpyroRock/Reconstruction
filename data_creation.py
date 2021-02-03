import numpy as np
from numpy import load, save
from PIL import Image
import matplotlib.pyplot as plt
from shuffle import shuffle_in_unison
import imageio
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from skimage import transform, io, img_as_ubyte


def read_data(rsp_size, im_size):
    #speckles_patterns_val = np.zeros((300, 500, 700))
    #for i in range(1, 301):
    #    speckles_patterns = loadmat(('C://Users//spime//OneDrive//Υπολογιστής//train//symbol_Responses//Symbol_response_'+str(k)+'_'+str(i)+'.mat'))
    #    speckles_patterns_val[i-1] = speckles_patterns['a2']
    dim_img = (im_size, im_size)
    dim_rsp = (rsp_size, rsp_size)
    for k in range(8, 11):
        for i in range(1, 259):
            speckle = loadmat('C://Users//spime//OneDrive//Υπολογιστής//train//symbol_Responses//Symbol_response_'+str(k)+'_'+str(i)+'.mat')
            speckle_val = speckle['a2']
            imageio.imwrite('speckle.jpg',speckle_val) 
            image = imageio.imread('speckle.jpg')
            # resize image
            resized = cv2.resize(image, dim_rsp, interpolation = cv2.INTER_CUBIC)
            #resized = imresize(image, dim, interp='bilinear')
            imageio.imwrite('C://Users//spime//OneDrive//Υπολογιστής//train//symbol_Responses_resized//'+str(k)+'//symbol_'+str(k)+'_'+str(i)+'.jpg', resized)

        for i in range(1, 259):
            symbol = imageio.imread('C://Users//spime//OneDrive//Υπολογιστής//train//'+str(k)+'//symbol_'+str(i)+'.jpg')
            # resize image
            resized = cv2.resize(symbol, dim_img, interpolation = cv2.INTER_CUBIC)
            #resized = transform.resize(symbol, dim, mode='symmetric', preserve_range=True)
            imageio.imsave('C://Users//spime//OneDrive//Υπολογιστής//train//'+str(k)+'_resized//symbol_'+str(k)+'_'+str(i)+'.jpg', resized)

def create_the_dataset(rsp_size, im_size):
    speckle_array = np.zeros((3, 258, rsp_size, rsp_size)) 
    symbol_array = np.zeros((3, 258, im_size, im_size))

    t=0
    for k in range(8, 11):
        for i in range(1, 259):
            speckle = Image.open('C://Users//spime//OneDrive//Υπολογιστής//train//symbol_Responses_resized//'+str(k)+'//symbol_'+str(k)+'_'+str(i)+'.jpg')
            speckle = np.asarray(speckle)
            speckle_array[t,i-1] = speckle
            symbol = Image.open('C://Users//spime//OneDrive//Υπολογιστής//train//'+str(k)+'_resized//symbol_'+str(k)+'_'+str(i)+'.jpg')
            symbol = np.asarray(symbol)
            symbol_array[t,i-1] = symbol
        t=t+1

    # speckle_array = speckle_array.astype(int)
    # symbol_array = symbol_array.astype(int)

    speckle_array = np.array(speckle_array, dtype=np.float32)
    symbol_array = np.array(symbol_array, dtype=np.float32)

    #speckle_array = np.array(speckle_array, dtype=np.uint8)
    #symbol_array = np.array(symbol_array, dtype=np.uint8)
    
    print(speckle_array.shape)
    print(symbol_array.shape)

    for i in range(speckle_array.shape[0]):
        speckle_array_case, symbol_array_case = shuffle_in_unison(speckle_array[i], symbol_array[i])
        save('speckle_array_case'+str(i)+'.npy', speckle_array_case)
        save('symbol_array_case'+str(i)+'.npy', symbol_array_case)
