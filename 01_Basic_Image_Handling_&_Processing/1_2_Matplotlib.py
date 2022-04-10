# Imports
from PIL import Image
from pylab import *
import numpy as np

# 1. Read image to array
image = array(Image.open("./resources/original/messi-ronaldo.jpeg"))

# 2. Plot the blank canvas
imshow(image)

# y axis starts from top left corner
# 3. adding some points to the chart
x = np.array[100,100,400,400]
y = np.array[200,500,200,500]

# 4. PLOT
 

plot(x,y,'r*') # points with red star-markers

# plot(x,y,'go-') # green line with circle markers

# plot(x,y,'ks:') # black dotted line with square-markers

# 5. LINE PLOT 

# connecting the first two points
plot(x[0:2],y[:2])

# More array slicing practicing
# Straight Lines
plot(x[0:4:2],y[0:4:2]) # x[first index:last index:interval steps]
plot(x[2:4],y[0:2])
plot(x[0:4:2],y[1:4:2])

# Diagonal Lines
plot(x[0:4:2],y[1:3])
plot(x[0:4:2],y[0:2])


# 6. Add title and show the plot
title('Plotting: "Messi Ronaldo" ')
show()
