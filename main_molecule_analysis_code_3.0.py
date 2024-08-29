import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

# Functions!
#####################################################################################################################
########################################


# this function calculates the gyration tensor on each frame of the movie
# and then returns the tensor and the radius of gyration calculated on the current frame
def calculate_gyration_tensor(img, xcom, ycom, Itot, xdim, ydim):
    """
    Function to calculate the gyration tensor and radius of gyration for a given frame.
    
    Parameters:
    img (ndarray): 2D array representing the intensity values of the frame.
    xcom (float): x-coordinate of the center of mass.
    ycom (float): y-coordinate of the center of mass.
    Itot (float): Total intensity of the frame.
    xdim (int): Width of the image (number of columns).
    ydim (int): Height of the image (number of rows).
    
    Returns:
    G (ndarray): 2x2 gyration tensor matrix.
    R_g (float): Radius of gyration.
    """

    gxx = 0.0
    gyy = 0.0
    gxy = 0.0

    for i in range(ydim):
        for j in range(xdim):
            I = img[i, j]
            gxx += I * (j - xcom) ** 2
            gyy += I * (i - ycom) ** 2
            gxy += I * (j - xcom) * (i - ycom)

    G = np.array([[gxx, gxy], [gxy, gyy]])
    G = G / Itot  # Normalize by total intensity

    eigenvalues, _ = np.linalg.eig(G)
    R_g = np.sqrt(eigenvalues[0] + eigenvalues[1]) 

    return G, R_g

# This function subtracts noise from the video of the particle, in order to better track it
# once noise is subtracted, calculates the center of mass of the particle and then 
# returns the x y coordinate of it and returns it
def track_particle(movie, initial_masking_value=3, bool_ex=1):
    """
    Function to track the center of mass of a particle in a multi-frame TIFF movie.
    Allows the user to adjust the masking value in a loop until satisfied, 
    then displays the movie with the tracked center of mass overlaid and returns the x and y coordinates.

    Parameters:
    movie (ndarray): The 3D numpy array representing the movie (frames, height, width).
    initial_masking_value (float): Initial value used for thresholding to isolate the particle. Default is 3.
    bool_ex (int): Flag to control whether to play the movie. Default is 1.

    Returns:
    xloc (ndarray): Array containing x-coordinates of the center of mass for each frame.
    yloc (ndarray): Array containing y-coordinates of the center of mass for each frame.
    """

    xdim, ydim = movie.shape[2], movie.shape[1]
    xloc = np.zeros(movie.shape[0])
    yloc = np.zeros(movie.shape[0])
    R_g_all = np.zeros(movie.shape[0])

    Intensity_size = np.zeros(movie.shape[0])
    avg_intensity = np.zeros(movie.shape[0])
    intensity_for_bp = np.zeros(movie.shape[0])

    continue_loop = True
    masking_value = initial_masking_value

    while continue_loop:
        plt.ion()  # Turn on interactive mode for real-time updating
        fig, ax = plt.subplots()

        for k in range(movie.shape[0]):
            testing_movie = movie[k, :, :].copy()
            average_intense = np.mean(testing_movie, axis=1)
            max_intense = np.max(average_intense)
            testing_movie[testing_movie >= max_intense] = 0

            intensities_no_zeroes = [
                testing_movie[ugh, ugh_2] for ugh in range(testing_movie.shape[0]) for ugh_2 in range(testing_movie.shape[1]) if testing_movie[ugh, ugh_2] != 0
            ]

            background_avg = np.mean(intensities_no_zeroes)

            img = movie[k, :, :] - background_avg
            movie_2 = img.copy()
            movie_3 = img.copy()

            movie_3 = gaussian_filter(movie_2, sigma=1)
            average_intense = np.mean(movie_3, axis=1)
            max_intense = np.max(average_intense)

            movie_3[movie_3 < max_intense + masking_value] = 0
            movie_3[movie_3 > 0] = 1

            img = movie_2 * movie_3
            # Store the sum of the intensity of the molecule on the current frame, which is proportional to its molecular weight
            intensity_for_bp[k] = np.sum(img)

            # working on this shit rn

            Itot = np.sum(img)
            Intensity_size[k] = Itot
            avg_intensity[k] = Itot
            xInt = np.sum(img, axis=0)
            yInt = np.sum(img, axis=1)
            xcom = np.sum(xInt * np.arange(0, xdim)) / Itot
            ycom = np.sum(yInt * np.arange(0, ydim)) / Itot
            xloc[k] = xcom
            yloc[k] = ycom


            # Calculate gyration tensor and radius of gyration
            G, R_g = calculate_gyration_tensor(img, xcom, ycom, Itot, xdim, ydim)
            R_g_all[k] = R_g  # Store the radius of gyration

            ax.imshow(img, cmap='gray')
            ax.plot(xloc[k], yloc[k], 'ro')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.0001)
            ax.clear()

        plt.ioff()
        plt.show()


        avg_intensity_for_bp = np.mean(intensity_for_bp)
        intensity_for_bp_uncertainty = np.std(intensity_for_bp, ddof = 1)



        # Ask the user if they are satisfied or want to adjust the masking value
        response = input(f"Current masking value is {masking_value}. Are you satisfied with the result? (yes to finish, no to adjust): ").strip().lower()
        if response == 'yes':
            continue_loop = False
        else:
            masking_value = float(input("Enter the new masking value: "))

    return xloc, yloc, R_g_all, avg_intensity_for_bp, intensity_for_bp_uncertainty
   
# This function calculates the mean squared displacement in the x, y and r directions
def msd_calculator(xloc, yloc):
    
    # based off of the x and y locations of particle, this function calculates the mean squared displacement via
    # Catipovic method 
    f = len(xloc)
    msd_x = np.zeros(movie.shape[0]) # mean squared displacement in horizontal direction
    msd_y = np.zeros(movie.shape[0]) # mean squared displacement in vertical direction
    msd_r = np.zeros(movie.shape[0]) # mean squared displacement in r direction (subtends x and y); just the sum of x and y mean squared displacement

    # nested for loop calculates msd_x...I like nested for loops lol
    
    for k in range(f):

        for j in range(f - k):
            squarey_x = ((xloc[k + j] -xloc[j])**2)/(f - k)
            msd_x[k] = msd_x[k] + squarey_x

    for k in range(f):

        for j in range(f - k):
            squarey_y = ((yloc[k + j] -yloc[j])**2)/(f - k)
            msd_y[k] = msd_y[k] + squarey_y

    
    for k in range(f):
        msd_r[k] = msd_x[k] + msd_y[k]

        

    return msd_x, msd_y, msd_r


# This function finds the diffusion via fitting to a portion of whatever mean squared displacement curve that the user wants
#  For example, MSD = ADt, where A =2 for 1d, A = 4 for 2d and A = 6 for 3d. Thus, MSD is linear in time
# So, fitting a line to MSD curve (the linear portion) can be used to find the diffusion value, since it's the slope of the fit line
# Will apply least linear squares to fit
def diffusion_finder(msd_x, msd_y, msd_r, frames):
    
    D = np.zeros(movie.shape[0])
    D_uncertainty = np.zeros(movie.shape[0])

    # Plotting msd_x, msd_y and msd_r here; make legend that says which curve is which yo

    plt.scatter(frames, msd_x, label='msdx', color='blue')
    plt.scatter(frames, msd_y, label='msdy', color='red')
    plt.scatter(frames, msd_r, label='msdr', color='green')

    plt.xlabel('Frames')
    plt.ylabel('Mean Squared Displacement (pixels^2)')

    plt.title('Mean Squared Displacement for x, y and r')
    plt.legend()
    plt.grid(True)
    plt.show()




    msd_choice = int(input("Enter 1 for msd_x, 2 for msd_y and 3 for msd_r: "))

    if msd_choice == 1:
        y = msd_x
        A = 2
    elif msd_choice == 2:
        y = msd_y
        A = 2
    elif msd_choice == 3:
        y = msd_r
        A = 4
    else:
        print("Woah, that wasn't an option!")
        return None, None

    
    while True:
        # Prompt the user to enter the range of frames for fitting
        initial_frame = int(input("Enter the initial frame index for fitting: "))
        final_frame = int(input("Enter the final frame index for fitting: "))

        # Slice the frames and MSD arrays to the selected range
        x_fit = frames[initial_frame:final_frame+1]
        y_fit = y[initial_frame:final_frame+1]

        # Plot the selected range
        plt.scatter(x_fit, y_fit, label=f'Selected range for fitting', color='purple')
        plt.xlabel('Frame Indices')
        plt.ylabel('Mean Squared Displacement (pixels^2)')
        plt.title('Selected Range for MSD Fitting')
        plt.grid(True)
        plt.show()

        # Confirm with the user if this range is correct
        response = input("Is this the range you want to fit? (yes to proceed, no to select again): ").strip().lower()
        if response == 'yes':
            break




    x = x_fit
    y = y_fit
    
    # ok, now to fit to this data set using linear least squares regression method......
    x_sum = sum(x)
    y_sum = sum(y)
    x_squared_sum = sum(x**2)
    x_y_sum = sum(x*y)
    N = len(x)

    slope = (N*x_y_sum - x_sum*y_sum)/(N*x_squared_sum -(x_sum)**2)
    y_intercept = (y_sum - slope*x_sum)/(N)

    # Finding uncertainties in fit parameters
    #uncertainty_y = np.sqrt((1/(N - 2))*(sum(y - slope - y_intercept*x))**2)
    #slope_uncertainty = uncertainty_y*np.sqrt(x_squared_sum/(N*x_squared_sum - (x_sum)**2))
    y_predicted = slope*x + y_intercept
    sum_residuals = np.sum((y - y_predicted)**2)
    sum_residuals = sum_residuals/(N - 2)
    cool_x = np.sum((x - np.mean(x))**2)
    slope_uncertainty = np.sqrt(sum_residuals/cool_x)
    

    # there's a problem with the uncertainty in slope damn
    # Finding Diffusion and Diffusion uncertainty
    D = slope/A
    D_uncertainty = slope_uncertainty/A 
    print(D)
    print(D_uncertainty)



    return D, D_uncertainty



#########################################################################################################################
##########################################







# Function calling and main work!
###############################################################################################################################
####################################################

fps = float(input("Enter the frames per second (fps): "))
pixel_conversion = float(input("Enter the microscope pixel conversion in nanometers: "))
bp_conversion = float(input("Enter the intensity calibration in bp to find molecular weight: "))

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Filename', 'Diffusion (um^2/s)', 'Diffusion Uncertainty (um^2/s)', 'Rg (um)', 'Rg_uncertainty (um)', 'Molecular Weight (bp)', 'Molecular Weight Uncertainty (bp)'])
msd_df = pd.DataFrame(columns = ['Filename', 'msd_x (um^2)', 'msd_y (um^2)', 'msd_r (um^2)' , 'time lag (sec)'])

#Put yo stuff here!!!

# Specify the directory containing TIFF files
directory = ''  # Adjust this path to your directory

# out to excel and pickle files for all data except for msd curves
output_excel = ''  # Specify the output Excel file; has to be .csv ....I know lol
output_pickle = ''  # Specify the output pickle file

# output to excel and pickle files for msd curves for molecules
output_excel_msd = ''  # Specify the output Excel file has to be .xlsx ... i know, weird lol
output_pickle_msd = ''  # Specify the output pickle file






# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        filepath = os.path.join(directory, filename)


        
        tif = tiff.TiffFile(filepath)
        movie = tif.asarray()  # Load the movie
        tif.close()

        frames = movie.shape[0]
        frame_indicies = np.arange(movie.shape[0])
        time_vec = np.arange(0,frames)/fps #converts frames to seconds based off of inputted frames per second

        xloc, yloc, R_g_all, intensity_for_bp, intensity_for_bp_uncertainty = track_particle(movie)

        msd_x, msd_y, msd_r = msd_calculator(xloc, yloc)
        D, D_uncertainty = diffusion_finder(msd_x, msd_y, msd_r, frame_indicies)

        # converting everything to real space units in micrometers and that
        xloc,yloc, R_g_all= xloc*(pixel_conversion/1000), yloc*(pixel_conversion/1000), R_g_all*(pixel_conversion/1000) #converts values to micrometers
        msd_x, msd_y, msd_r = msd_x*(pixel_conversion / 1000)**2. ,msd_y*(pixel_conversion / 1000)**2, msd_r*(pixel_conversion / 1000)**2
        D, D_uncertainty = D*fps*(pixel_conversion / 1000)**2, D_uncertainty*fps*(pixel_conversion / 1000)**2

        # Finding radius of gyration and its uncertainty 
        R_g_average = np.mean(R_g_all)
        R_g_uncertainty = np.std(R_g_all, ddof = 1)

        #converting intensities to find molecular weight in bp
        molecular_weight = intensity_for_bp*bp_conversion
        molecular_weight_uncertainty = intensity_for_bp_uncertainty*bp_conversion


        # Creating dataframe with the data

        new_row = pd.DataFrame({
            'Filename': [filename],
            'Diffusion (um^2/s)': [D],
            'Diffusion Uncertainty (um^2/s)': [D_uncertainty],
            'Rg (um)': [R_g_average],
            'Rg_uncertainty (um)': [R_g_uncertainty],
            'Molecular Weight (bp)': [molecular_weight],
            'Molecular Weight Uncertainty (bp)': [molecular_weight_uncertainty]
        })

        # Concatenate the new row to the results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)


        


       

        # for msd stuff
        new_msd_row = pd.DataFrame({'Filename: ': [filename]*len(msd_x),'msd_x (um^2)': msd_x, 'msd_y (um^2)': msd_y, 'msd_r (um^2)': msd_r, 'time lag (sec)': time_vec})
        new_msd_row = new_msd_row.T
    

        msd_df = pd.concat([msd_df, new_msd_row], ignore_index = True)




        # Save DataFrame to CSV
        results_df.to_csv(output_excel, index=False)

        # Save DataFrame to a pickle file
        results_df.to_pickle(output_pickle)


        msd_df.to_csv(output_excel_msd, index = False)
        msd_df.to_pickle(output_pickle_msd)
        

################################################################################################################################################################
######################################################################

