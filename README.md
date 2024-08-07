# Image Analysis Codes, Python
 These are python version of codes that I wrote to analyze images and videos, that I took via fluorescence, in my research in grad school.



 ~ Codes in this repository ~

 1.) Basic Tiff reader and 'movie' player





 ~ Description of codes ~

 1.) Basic Tiff reader and 'movie' player: This Python code reads a multi-frame TIFF file, stores the pixel data of each frame in a three-dimensional numpy array named "movie," and then plays the frames as a movie using matplotlib. It handles frames of varying sizes by determining the maximum height and width of all frames, allocating the "movie" array based on these dimensions, and padding (adding pixels with zero intensity) smaller frames to fit. The code uses the tifffile library to read the TIFF file and matplotlib to display the frames sequentially, creating a movie effect.
