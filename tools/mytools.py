from PIL import Image
import numpy as np

#-----------------------------------------------------------------------#

# 3. IMAGE RESIZING

# Image resizing function 
def imresize(file,Size):
    """Resize an image array using PIL. """
    image = Image.fromarray(np.uint8(file))
    
    return np.array(image.resize(Size))

#-----------------------------------------------------------------------#

# 4. HISTOGRAM EQUALIZATION

def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True) 
    cdf = imhist.cumsum() # cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape), cdf

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