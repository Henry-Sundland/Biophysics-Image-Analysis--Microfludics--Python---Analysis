import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the TIFF file
filename = '' #place path to file here
tif = tiff.TiffFile(filename)
movie = tif.asarray()
tif.close()

# Assuming 'masking_value' and 'bool_ex' are defined elsewhere in your code
masking_value = 3  # Adjust this value as needed
bool_ex = 1  # Adjust based on your requirements

# Initialize the plot
plt.ion()  # Turn on interactive mode for real-time updating
fig, ax = plt.subplots()

# Initialize the plot image object once
image_plot = ax.imshow(movie[0, :, :], cmap='gray')
plt.show(block=False)


# Loop through each frame
for k in range(movie.shape[0]):
    # Average background test for frame
    testing_movie = movie[k, :, :].copy()  # Equivalent to movie(:,:,k) in MATLAB
    average_intense = np.mean(testing_movie, axis=1)  # Adjusted to operate along the correct axis
    max_intense = np.max(average_intense)
    testing_movie[testing_movie >= max_intense] = 0

    # Loop through frame and save all intensities that are not zero
    t = 0
    intensities_no_zeroes = []
    for ugh in range(testing_movie.shape[0]):
        for ugh_2 in range(testing_movie.shape[1]):
            dummy = testing_movie[ugh, ugh_2]

            if dummy != 0:
                intensities_no_zeroes.append(dummy)
                dummy = 0
                t += 1
            dummy = 0

    # Calculate the background average
    background_avg = np.mean(intensities_no_zeroes)

    # Noise subtraction of current movie iteration
    img = movie[k, :, :] - background_avg
    movie_2 = img.copy()
    movie_3 = img.copy()

    # Apply Gaussian filter
    movie_3 = gaussian_filter(movie_2, sigma=1)
    average_intense = np.mean(movie_3, axis=1)  # Ensure correct axis usage
    max_intense = np.max(average_intense)

    # Thresholding
    movie_3[movie_3 < max_intense + masking_value] = 0
    movie_3[movie_3 > 0] = 1

    # Apply mask
    img = movie_2 * movie_3

   # Update the image data in the plot
    image_plot.set_array(img)
    fig.canvas.draw()  # Draw the figure
    fig.canvas.flush_events()  # Flush any pending GUI events
    plt.pause(0.001)  # Pause to allow real-time update

plt.ioff()  # Turn off interactive mode
plt.show()  # Ensure the last frame remains visible until the figure is closed

   




