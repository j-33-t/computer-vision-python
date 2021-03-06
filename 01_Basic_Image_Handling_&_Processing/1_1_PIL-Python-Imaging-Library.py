# Imports
from PIL import Image

from matplotlib.image import thumbnail

# 1. Read Image
img  = Image.open("./resources/original/messi-ronaldo.jpeg")

# Image object
img

# Example of Image object atributes
img.info
img.height
img.width
img.__class__

# Display Image
img.show()

# Color conversion (Convert to Grayscale)

imgGrayscale = img.convert('L')
imgGrayscale.show()

# 2. Save to another format (jpg,png)
imgGrayscale.save("./resources/exports/messi-ronaldo_GrayScale.jpg")
imgGrayscale.save("./resources/exports/messi-ronaldo_GrayScale.png")


# 3. Create Thumbnails
img.thumbnail((128,128)) # 128 pixels
img.show()


# 3. Copy and Paste Regions
img = Image.open("./resources/exports/messi-ronaldo_GrayScale.jpg")
box = (100,100,400,400) # (left, upper, right, lower)
region = img.crop(box)
region = region.transpose(Image.ROTATE_180)
img.paste(region,box)
img.show()

# 4. Resize and Rotate

# 4.1 resize
img = Image.open("./resources/exports/messi-ronaldo_GrayScale.jpg")
out = img.resize((128,128))
out.show()

# 4.2 rotate
out = img.rotate(45)
out.show()

