# Image Analysis Codes, Python
 These are python version of codes that I wrote to analyze images and videos, that I took via fluorescence, in my research in grad school.



 ~ Codes in this repository ~

 1.) Basic Tiff reader and 'movie' player
 
 2.) Tiff reader and 'movie' player that subtracts noise

 3.) Particle Tracking 

 4.) Main Analysis Code





 ~ Description of codes ~

 1.) Basic Tiff reader and 'movie' player: This Python code reads a multi-frame TIFF file, stores the pixel data of each frame in a three-dimensional numpy array named "movie," and then plays the frames as a movie using matplotlib. It handles frames of varying sizes by determining the maximum height and width of all frames, allocating the "movie" array based on these dimensions, and padding (adding pixels with zero intensity) smaller frames to fit. The code uses the tifffile library to read the TIFF file and matplotlib to display the frames sequentially, creating a movie effect.

 2.) Tiff reader and 'movie' player that subtracts noise code processes each frame of a multi-frame TIFF file by subtracting background noise using the average intensity of non-zero pixels, applying a Gaussian filter to smooth the image and reduce noise, and creating a boolean mask to isolate significant features based on a threshold. The masked image highlights only the important areas of the frame, while the rest is suppressed. The code then plays these processed frames as a continuous movie in a single figure window, ensuring smooth playback and clear visualization of the enhanced features. The purpose of this being to better track a particle via particle tracking methods. In the code, enter your own threshold value to adjust the threshold that noise is eliminated (a greater value will subtract more noise, so play around with it), and bool_ex being set to a value of 1 just plays the movie. Set it to 0 if you don't want the movie to play. The noise subtracted matrix, for each k frame, is saved as the variable 'img'.

 3.) Particle Tracking code processes a multi-frame TIFF video to track the center of mass of a particle across all frames. It begins by loading the video and initializing variables to store the x and y coordinates of the particle's center of mass for each frame. The code then iterates through each frame, applying noise subtraction and a Gaussian filter to enhance the particle's visibility. A thresholding step creates a mask to isolate the particle, and the center of mass is calculated based on the intensity distribution. Finally, the code overlays a red dot on the particle's location in each frame and displays the video with this tracking, allowing visual confirmation that the calculated coordinates correspond to the particle's actual position throughout the video.



4.) This code is designed to analyze movies of particles or molecules, represented as a series of frames in TIFF files, and extract various physical properties such as the diffusion coefficient, radius of gyration, and molecular weight. It begins by loading TIFF files from a specified directory, each representing a movie of a particle or molecule under observation. The track_particle function is used to subtract background noise and track the center of mass of the particle in each frame. This function allows the user to adjust a masking value to isolate the particle, and it returns the x and y coordinates of the center of mass for each frame, along with the average intensity values. These intensity values are later used to calculate properties like the radius of gyration. For each frame, the code calculates the gyration tensor using the calculate_gyration_tensor function, which determines the radius of gyration, a measure of the particle's spread around its center of mass. The radius of gyration is calculated for each frame and then averaged over the entire movie to provide a stable measure of the particle's size in space.

Next, the msd_calculator function is used to calculate the mean squared displacement (MSD) of the particle in the x, y, and r directions. The MSD is a measure of the particle's movement over time and is essential for determining the diffusion coefficient. The function employs nested loops to calculate the MSD by considering the displacement of the particle over different time intervals. Once the MSD values are obtained, the diffusion_finder function fits a linear model to the MSD data to find the diffusion coefficient, which is the slope of the linear portion of the MSD curve. This function allows the user to select which MSD curve to fit and the range of frames to use for the fitting process. It then calculates the slope and intercept of the fit, along with the uncertainty in the slope, which is used to determine the uncertainty in the diffusion coefficient.

Finally, the code converts the calculated values into real space units, such as micrometers for distances and square micrometers per second for diffusion coefficients. It also calculates the molecular weight based on the intensity values and their calibration. All these calculated properties, including their uncertainties, are stored in a pandas DataFrame, which is then saved as an Excel file for further analysis. The result is a detailed analysis of the particle's movement and physical properties, all derived from the movie data provided in the TIFF files.