import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import ImageTk, Image 
import os
import scipy
import scipy.optimize as opt
import math as m
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import cv2
import matplotlib.pyplot as plt
import glob

class cal_point_selection(ctk.CTk):
    def __init__(self):
        super().__init__()

        # title window
        self.title('Calibration')                                                                                      

        # initialize counters and output coordiante matrix
        self.objects = []
        self.coords = np.zeros(shape=(50, 5))
        self.i = 0

        # Get file path for calibration image from user and 
        # open it into a window (the size of the window is the size of the  
        # image such that the pixel clicked on on the image is the same as the 
        # pixel on the sensor, this need to be made more robust later)
        
        self.path = tk.filedialog.askopenfilename()
        self.cal_img = ImageTk.PhotoImage(Image.open(self.path))
        img_dims = str(self.cal_img.width()) + 'x' + str(self.cal_img.height())
        self.window = tk.Canvas(self)
        self.window.pack(side = 'bottom', fill = 'both', expand = 'yes', padx=0, pady=0)
        self.window.create_image((self.cal_img.width()/2, self.cal_img.height()/2), image = self.cal_img)

        self.geometry(img_dims)

        # Binding buttons to actions
        self.window.bind('<Button-1>', self.Lclick)
        self.window.bind('<Button-3>', self.Rclick)
        self.bind('<Return>', self.enter)

    # Mark a calibration point using Left click
    def Lclick(self, event):
        print('clicked at x = % d, y = % d'%(event.x, event.y))
        self.object = self.window.create_oval(event.x - 3, event.y + 3, event.x + 3, event.y - 3, outline = 'red', fill = 'red')
        self.objects.append(self.object)

        dialog = ctk.CTkInputDialog(text = 'Input Point coords in the form of: x,y,z', title = 'Input')

        try:
            temp_coords = [float(coord) for coord in dialog.get_input().split(',')]
            if len(temp_coords) != 3:
                raise ValueError("Please enter exactly three coordinates.")
        except (ValueError, TypeError):
            print("Invalid input. Please enter three integer coordinates separated by commas.")
            self.window.delete(self.object)
            self.objects.pop()
            return

        print(temp_coords)
        self.coords[self.i,:] = [event.x, event.y, temp_coords[0], temp_coords[1], temp_coords[2]]
        self.i += 1
       
    # Delete the last calibration point    
    def Rclick(self, event):
        print('Last point deleted')
        last_object = self.objects.pop()
        self.window.delete(last_object)
        self.i -= 1
        
    # Once done you can press enter to exit the calibraion window
    def enter(self, event):
        self.destroy()
        
        self.coords = self.coords[0:self.i,:]
        print('calibration points set: ')
        np.save('coords.npy', self.coords)
        print(self.coords)
        return self.coords

class cal:
    def __init__(self, coords):
        
        if coords[:,0].shape[0] < 6: raise ValueError('Not enough observation points, 6 or more required')

        uv, xyz = np.split(coords, (2,), axis = 1)

        self.A = np.zeros((2*coords.shape[0],12))

        for i in range(coords.shape[0]):
            self.A[i*2:i*2 + 2,:] = np.vstack((np.concatenate([xyz[i,:], np.ones(1), np.zeros((4)), -uv[i,0]*np.hstack((xyz[i,:], 1))]),
                                               np.concatenate([np.zeros((4)), xyz[i,:], np.ones(1), -uv[i,1]*np.hstack((xyz[i,:], 1))])))

        
        eig_val, eig_vec = np.linalg.eig(self.A.T @ self.A)
        min_eig = np.argmin(eig_val)
        self.p = eig_vec[:,min_eig].reshape((3,4))
        
        _, s, v_t = np.linalg.svd(self.A)
        print(self.p)
        self.p2 = v_t.T[:,-1].reshape((3,4))
        # print(self.p2)
        # self.p = self.p2

       
        # s = -1
        # A_bar = np.sign(np.linalg.det(self.p[:,:3]) * self.p[:,:3])
        # A_bar_inv = np.linalg.inv(A_bar)
        # R_bar_t, k_bar_inv = np.linalg.qr(A_bar_inv)
        # R_bar = R_bar_t.T; k_bar = np.linalg.inv(k_bar_inv)
        # D = np.diag(np.sign(np.diag(k_bar))) @ np.diag(np.array([s, s, 1]))
        # self.k = k_bar @ D; self.k /= self.k[-1,-1]
        # self.rot_mat = D @ R_bar

        R_inv, k_inv = np.linalg.qr(np.linalg.inv(self.p[:,:3]))

        self.rot_mat = R_inv.T @ np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
        print(np.linalg.det(self.rot_mat))
        self.k = np.linalg.inv(k_inv)
        self.k /= self.k[-1,-1]

        # self.k, self.rot_mat = scipy.linalg.rq(self.p[:,:3])
        # self.k /= self.k[-1,-1]

        self.translation_vec = -np.linalg.inv(self.p[:,:3]) @ self.p[:,-1]

        print('Camera matrix is:')
        print(self.k)
        print('Rotation matrix is:')
        print(self.rot_mat)
        print('Camera postion is:')
        print(self.translation_vec)

    def get_uv(self, xyz: np.ndarray) -> np.ndarray:

        uv_homg = self.p @ np.hstack((xyz, np.ones_like(xyz[:,0:1]))).T
        
        uv_norm = uv_homg / uv_homg[2,:]

        print('u, v are:')
        print(uv_norm[:-1,:])

        return uv_norm

    def get_xyz(self, uv, zc: float = 1):
        # uv[:,:] *= 1
        uv_homg = np.hstack((uv, np.ones_like(uv[:,0:1]))).T
        # print(uv_homg)

        k_inv = np.linalg.inv(self.k)
        # print(k_inv)

        xyz_c = k_inv @ uv_homg
        # print(xyz_c)

        xyz_c *= zc

        # xyz = (self.rot_mat.T @ xyz_c).T - (self.rot_mat.T @ self.T)
        # print(xyz)
        xyz = self.rot_mat.T @ (xyz_c - self.translation_vec[np.newaxis,:].T)
        print('x, y, z via k, r and t are:')
        print(xyz)
        # xyz = (self.rot_mat.T @ xyz_c).T - (self.rot_mat.T @ self.T)
        # print(xyz)

        xyz = np.linalg.pinv(self.p) @ uv_homg * zc 
        xyz /= xyz[-1,0]

        print('x, y, z via pinv are:')
        print(xyz)

        xyz =  self.translation_vec[np.newaxis,:].T + zc * np.linalg.inv(self.k @ self.rot_mat) @ uv_homg
        
        return xyz

class calibrate:
    def __init__(self,
                 cal_coords: np.ndarray,
                 folder_path: str = 'cal_img_2',
                 units: str = 'mm',
                #  sensx: float = 5.6448,
                 sensx: float = 23.5,
                 chessboard_shape: tuple[int, int] = (7,10),  
                 chessboard_size: float = 25
) -> None:
        
        self.coords = cal_coords    
        
        # stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed. 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
        
        # List for 3D points 
        world_points = [] 
        
        # List for 2D points 
        image_points = [] 
        
        #  3D points real world coordinates in dummy coordinates
        corner_coords = np.zeros((1, chessboard_shape[0]  * chessboard_shape[1], 3), np.float32) 
        corner_coords[0, :, :2] = (chessboard_size * np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]]).T.reshape(-1, 2)
        
        # Extracting path of individual image stored in a given directory. Since no path is specified, it will take current directory  jpg files alone 
        images = glob.glob(folder_path + '/*.jpg') 
        
        for filename in images: 
            image = cv2.imread(filename) 
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            
            # Find the chess board corners If desired number of corners are found in the image then ret = true 
            ret, corners = cv2.findChessboardCorners( grayColor, chessboard_shape,  
                                                      cv2.CALIB_CB_ADAPTIVE_THRESH  + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 

            # If desired number of corners can be detected then, refine the pixel coordinates and display them on the images of checker board 
            if ret == True: 
                world_points.append(corner_coords) 
        
                # Refining pixel coordinates for given 2d points. 
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 
        
                image_points.append(corners2) 

                # Uncomment the following 4 lines to display the corners found: 
                # image = cv2.drawChessboardCorners(image,  chessboard_shape, corners2, ret) 
        
        #     cv2.imshow('img', image) 
        #     cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
    
        
        # Perform camera calibration by passing the value of above found out 3D points (world_points) and its corresponding pixel coordinates of the 
        # detected corners (image_points) 
        rep_error, self.cam_mat, self.dist_coeffs, _, _ = cv2.calibrateCamera(world_points, image_points, grayColor.shape[::-1], None, None)
        
        print('-'*60,'\n')
        np.set_printoptions(suppress=True,precision= 3)
        print(" Camera matrix: \n", np.round(self.cam_mat, 2)) 
        
        print("\n Distortion coefficients: \n", self.dist_coeffs) 

        image_points, world_points = np.split(cal_coords, (2,), axis = 1)
        image_points = image_points.astype('float32'); world_points = world_points.astype('float32')

        ret, self.rvec, self.tvec = cv2.solvePnP(world_points, image_points, self.cam_mat, self.dist_coeffs)
        # self.rvec, self.tvec= cv2.solvePnPRansac(world_points, image_points, self.cam_mat, self.dist_coeffs)
        
        self.rot_mat, _ = cv2.Rodrigues(self.rvec)
        self.rot_mat = self.rot_mat.T

        self.translation_vec = - self.rot_mat @ self.tvec

        print('\n Rotation matrix: \n', self.rot_mat)       

        print('\n Translation vector (' + units + '): \n', self.translation_vec)

        imgx = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE).shape[1]
        mx = sensx/imgx
        f = self.cam_mat[1,1]*mx
        print('Focal length is:', round(f, 2), units)

        img_points_projected, _ = cv2.projectPoints(world_points, self.rvec, self.tvec, self.cam_mat, self.dist_coeffs)
        img_points_projected = img_points_projected[:,0,:]

        reprojection_error = np.mean(np.linalg.norm(image_points - img_points_projected, axis=1))

        print('Reprojection error from intrinsic calibratrion:', round(rep_error,2), 'pixels \n'
              ,'Reprojection error from extrinsic calibration:', round(reprojection_error,3), 'pixels')
        
        print('-'*60,'\n')


    def get_xyz(self,
                image_coords: np.ndarray,
                zc: float = 1,
                should_print: bool = False
    ) -> np.ndarray:
        if np.all(image_coords == None): return np.array([[None,None,None]]).T
        # Undistord image points 
        image_coords = np.squeeze(cv2.undistortImagePoints(image_coords.astype('float32'), self.cam_mat, self.dist_coeffs))


        if len(image_coords.shape) == 1: image_coords = image_coords[np.newaxis,:]

        uv_homg = np.hstack((image_coords, np.ones_like(image_coords[:,0:1]))).T

        k_inv = np.linalg.inv(self.cam_mat)

        xyz_c = k_inv @ uv_homg

        xyz_c *= zc

        xyz = self.rot_mat @ xyz_c + self.translation_vec
    
        if should_print == True: print('x, y, z  are: \n', xyz)
        return xyz
    
    def get_uv(self,
               world_coords: np.ndarray,
    ) -> np.ndarray:
        return cv2.projectPoints(world_coords, self.rvec, self.tvec, self.cam_mat, self.dist_coeffs)[0][:,0,:]

class PolynomialMapping:
     # all calibration points have to have x_p & y_p from all cameras
    def __init__(self, coords, poly_degree, nb_cameras):
        self.coords      = coords
        self.poly_degree = poly_degree
        self.nb_cameras  = nb_cameras

        new_coords = np.zeros((self.coords.shape[0], self.nb_cameras*2 + 3))
        new_coords[:, 0:3] = self.coords[:, 0:3]; new_coords[:, -3:] = self.coords[:,2:5] 
        indicies   = np.zeros((self.coords.shape[0], int(self.coords.shape[1]/5 - 1)))

        for i in range(1, int(self.coords.shape[1]/5 - 0)):
            for j in range(self.coords.shape[0]):
                for k in range(self.coords.shape[0]):
                    if (self.coords[j,2:5] == self.coords[k,2 + 5*i:5 + 5*i]).all(): 
                        indicies[j,i - 1] = k; 
                        new_coords[j, 2*i : 2 + 2*i] = self.coords[k, 5*i : 2 +5*i]

        self.coords = new_coords                

        np.random.shuffle(self.coords)
        X = self.coords[:,0:-3]
        Y = self.coords[:,-3:]

        a       = int(X.shape[0]*0.8)
        X_learn = X[:a,:]
        Y_learn = Y[:a,:]

        X_val   = X[a+1:,:]
        Y_val   = Y[a+1:,:]

        self.poly = PolynomialFeatures(degree = self.poly_degree)

        X_learn_ = self.poly.fit_transform(X_learn)

        self.Xmapfunc = linear_model.LinearRegression(); self.Ymapfunc = linear_model.LinearRegression(); self.Zmapfunc = linear_model.LinearRegression()
        self.Xmapfunc.fit(X_learn_, Y_learn[:,0])
        self.Ymapfunc.fit(X_learn_, Y_learn[:,1])
        self.Zmapfunc.fit(X_learn_, Y_learn[:,2])

        sum = 0

        for i in range(X_val.shape[0]):
            sum   = sum   + np.transpose(np.asmatrix(Y_val[i,:])) - (np.matrix([self.Xmapfunc.predict(self.poly.fit_transform([X_val[i,:]])), self.Ymapfunc.predict(self.poly.fit_transform([X_val[i,:]])), self.Zmapfunc.predict(self.poly.fit_transform([X_val[i,:]]))]))
        

        # mean_error_norm = 1/(X_val.shape[0])*sum3d
        self.mean_error_vec  = 1/(X_val.shape[0])*sum
        self.mean_error_norm = np.linalg.norm(self.mean_error_vec)   
        
    def map(self, coords):
         X_ = self.poly.fit_transform(coords)
         return np.array([[self.Xmapfunc.predict(X_)],[self.Ymapfunc.predict(X_)],[self.Zmapfunc.predict(X_)]])