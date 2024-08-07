import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Load the TIFF file
filename = '' #place path to file here
# Load the TIFF file
tif = tiff.TiffFile(filename)

# Get the dimensions of the image
images = tif.asarray()
frames = images.shape[0]
max_height = max(image.shape[0] for image in images)
max_width = max(image.shape[1] for image in images)

# Create the movie matrix with maximum dimensions
movie = np.zeros((max_height, max_width, frames), dtype=np.uint16)

# Populate the movie matrix with the image data, adjusting frame sizes if necessary
for i in range(frames):
    height, width = images[i].shape
    movie[:height, :width, i] = images[i]

# Close the TIFF file
tif.close()

# Display the movie frames
plt.ion()  # Turn on interactive mode for real-time updating
fig, ax = plt.subplots()
for i in range(frames):
    ax.clear()
    ax.imshow(movie[:, :, i], cmap='gray')
    plt.draw()
    plt.pause(0.001)  # Pause to create a movie effect

plt.ioff()  # Turn off interactive mode
plt.show()