This is a program that can be used to process raw PTV data to obtain particle tracks.

High-Level overview of how the code works:

  1) Camera Calibration:
     - For each camera take ~10 images (jpgs) of the chessboard pattern at varying angles ensuring that the chessboard fills the majority of the frame
     - Save theses images in folders corresponding to each camera
     - Take images of the target that will define the world coordinate frame with each camera in position
     - These images can then be passed to the code and the cameras calibrated
     - Finally the images of the experiment from each camera should be named sequentially, placed in folders corresponding to each camera
    
  2) Particle Detection and tracking:
      - Particles in each image are detected using a blob detector whose paremeters can be tuned
      - For each detected particle at each time step a ray is drawn and these rays are then searched for intersections (within a user defined tolerance) and finally the intersections are clustered to obtain particles
      - This is done at each timestep to give a particle distribution for each frame
      - The particles are then tracked by finding correspondences between particles at each frame
      - This is done one a frame by frame basis with the core method used to find correspondences being finding the nearest neighbour within some user defined theshold
      - Once several past particle positions have been determined a prediction step can be added where the position of the particle in the next frame is estimated using polyniomial extrapolation or weiner filter              prediction and the nearest neighbour of the estimated position is taken the be the next particle position
      - Iterating over each frame will then produce a set of particle tracks and finally the tracks can be searched to find and rejoin any tracks that have been broken into multiple tracks
    
  3) Data Formats and other stuff:
       - Cameras are defined in the code as instances of a camera class which contains all the calibration data ect.. and critically the class method that projects from image coordinates to world coordinates. All the          cameras are then stored is a list.
       - The image particle detection returns Nx2 arrays of row vectors containing the image (pixel) coordinates of the N detected particles. All the particle postions from each camera and each frame is then stored            in a list of lists of arrays:
             [[p(t0) ..... p(tn)]_cam0 ..... [p(t0) ..... p(tn)]_camk] w/ [] = list & p(t) is the Nx2 np.array of particle positions with n frames and k cameras
       - This can then be put in the form of:
           [[p_cam0 ... p_camk]_t0 ... [p_cam0 ... p_camk]_tn]
         Such that the image coordinates can be projected from each camera and the particle positions triangulated
       - The particle distribution at each frame is then saved as a list of Nx3 arrays where each row vector is a particle position and each list element (array) correspondes to an image frame.
       - The tracks are saved as a list of arrays with each array corresponding to particle track, each array is Nx4 with each row a timestamp followed by the particle coordinates. 
  
