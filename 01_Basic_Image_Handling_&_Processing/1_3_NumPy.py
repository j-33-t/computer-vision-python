# Imports

from PIL import Image
import numpy as np
import pandas as pd
from pylab import *
import glob

# 1. Array Image Representation
im = np.array(Image.open('./resources/original/messi-ronaldo.jpeg'))
print(im.shape,im.dtype) # shape(rows,columns,color channels) #uint8 = 8 bit integer

print(im)
print(im.ndim) # no of dimensions

im2 = np.array(Image.open('./resources/original/messi-ronaldo.jpeg').convert('L'),'f')
print(im2.shape,im2.dtype)


# Accessing elemets in = "im" array
im[0] # im[i,j,k] -- i,j = coordinates k = color channel
im[0,0]
im[0,0,0]
im[0,0,1]
im[0,0,2]

#-----------------------------------------------------------------------#

# 2. GRAYLEVEL TRANSFORMS
im3 = 255 - im2 # Invert Image color (0 - 255)
im4 = (100.0/255) * im2 + 100 # clamp to interval 100..200
im5 = 255.0 * (im2/255.0)**2 # Squared

# Reverse Grayscale array to Image

#im3 
imageTransform_im3 = Image.fromarray(im3)
imageTransform_im3.show()


# """
# If you did some operation to change the type from “uint8” 
# to another data type, such as im4 or im5 in the example above, 
# you need to convert back before creating the PIL image
# """

#im4 
imageTransform_im4 = Image.fromarray(np.uint8(im4))
imageTransform_im4.show()

#im5 
imageTransform_im5 = Image.fromarray(np.uint8(im5))
imageTransform_im5.show()


#-----------------------------------------------------------------------#

# 3. IMAGE RESIZING

# Image resizing function 
def imresize(file,Size):
    """Resize an image array using PIL. """
    image = Image.fromarray(np.uint8(file))
    
    return np.array(image.resize(Size))

function_test = Image.fromarray(imresize(im, (300,200)))
function_test.show()

#-----------------------------------------------------------------------#

# 4. HISTOGRAM EQUALIZATION

# function for histogram equalization
def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True) 
    cdf = imhist.cumsum() # cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape), cdf

# Testing function
im = np.array(Image.open("./resources/exports/messi-ronaldo_GrayScale.jpg"))
functionTest2 ,cdf = histeq(im)
result =  Image.fromarray(np.uint8(functionTest2))
result.show() # Contrast appear clearly


#-----------------------------------------------------------------------#

# 5. Averaging Images
def compute_average(imlist):
    
    """ Compute the average of a list of images. """
    
    # open first image and make into array of type float
    averageim = np.array(Image.open(imlist[0:]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '...skipped') 
        averageim /= len(imlist)
    # return average as uint8
    return np.array(averageim, 'uint8')

# Testing Function

path = ('./resources/exports/messi-ronaldo_GrayScale.jpg')

result = Image.fromarray(compute_average(path))
result.show()

#-----------------------------------------------------------------------#

# PRINCIPAL COMPONENT ANALYSIS (PCA) of Images

def pca(X):
    """ Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows return: projection matrix (with important dimensions first), variance and mean."""
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0) 
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = np.dot(X,X.T) 
        # covariance matrix
        e,EV = np.linalg.eigh(M) 
        # eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want 
        S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order 
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
        
    # return the projection matrix, the variance and the mean
    return V,S,mean_X


# Testing Function

# Step 1 --------
# Store all images objects as list 

imlist = []
for filename in glob.glob('./resources/a_selected_thumbs/*'):
    im=Image.open(filename)
    imlist.append(im)

# Step 2 --------
im = array(imlist[0]) # open one image to get size 
m,n = im.shape # get the size of the images
imnbr = len(imlist) # get the number of images
# create matrix to store all flattened images
immatrix = array([array(im).flatten() for im in imlist],'f')
# perform PCA
projection_matrix,variance,mean1 = pca(immatrix)
# show some images (mean and 7 first modes)
figure()
gray()
subplot(2,4,1) 
imshow(mean1.reshape(m,n)) 
for i in range(7):
    subplot(2,4,i+2) 
    imshow(projection_matrix[i].reshape(m,n))
show()
