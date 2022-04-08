# Imports
from PIL import Image
from pylab import *

# Read image to array
image = array(Image.open("./resources/exports/messi-ronaldo_GrayScale.png"))

# Plot the image
imshow(image)

# y axis starts from top left corner
# some points
x = [100,100,400,400]
y = [200,500,200,500]

# plot the points with read star-markers
plot(x,y,'r*')

# Line plot connecting the first two points
plot(x[:2],y[:2])

# Add title and show the plot
title('Plotting: "Messi Ronaldo" ')
show()
