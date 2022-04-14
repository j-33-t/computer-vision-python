from PIL import Image
from numpy import *
from scipy.ndimage import filters

# Load Image and convert to array
im = array(Image.open("./resources/original/messi-ronaldo.jpeg").convert('L'))

# Apply Blur on Image with standard deviation = 5 
im2 = filters.gaussian_filter(im,1)

# Convert array to image
result = Image.fromarray(im2)

# Display Result
result.show()

# Applying different level of blur 
im3 = filters.gaussian_filter(im,2)
im4 = filters.gaussian_filter(im,3)
im5 = filters.gaussian_filter(im,4)
result2 , result3 , result4 =  Image.fromarray(im3) ,Image.fromarray(im4),Image.fromarray(im5)
result2.show()
result3.show()
result4.show()