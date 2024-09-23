This is a program that can be used to process raw PTV data to obtain particle tracks.

High-Level overview of how the code works:

  1) Camera Calibration:
     - For each camera take ~10 images (jpgs) of the chessboard pattern at varying angles ensuring that the chessboard fills the majority of the frame
     - Save theses images in folders corresponding to each camera
     - Take images of the target that will define the world coordinate frame with each camera in postition
     - These images can then be passed to the code and the cameras calibrated
     - Finally the images of the experiment from each camera should be named sequentially, placed in folders corresponding to each camera
    
  2) Particle Detection and tracking:
      - Particles in each image are detected using a blob detector whose paremeters can be tuned
      - For each detected particle at each time step a ray is drawn and these rays are then searched for intersections (within a user defined tolerance) and finally the intersections are clustered to obtain particles
      - This is done at each timestep to give a particle distribution for each frame
      - 

  
