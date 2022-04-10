from PIL import Image
import numpy as np
import pandas as pd

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