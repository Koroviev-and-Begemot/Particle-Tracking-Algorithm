import tkinter as tk
from tkinter import filedialog, scrolledtext
import customtkinter as ctk
from PIL import ImageTk, Image 
import sys
import numpy as np
import pandas as pd

from calibration import*
from tracking import*

# Initialize the CustomTkinter theme
ctk.set_appearance_mode('dark')  
ctk.set_default_color_theme('green')  

class RedirectText:
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.config(state='normal')
        self.output.insert(tk.END, string)
        self.output.config(state='disabled')
        self.output.see(tk.END)

    def flush(self):
        pass  

class ConsoleWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Console")
        self.geometry("600x400")

        self.console_text = scrolledtext.ScrolledText(self, state='disabled', height=10, bg="black", fg="white")
        self.console_text.pack(pady=5, padx=10, fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.hide_console)
        self.hide_console()

    def hide_console(self):
        self.withdraw()

class CalPointSelection(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)

        # title window
        self.title('Calibration')                                                                                      

        # initialize counters and output coordiante matrix
        self.objects = []
        # self.coords = np.zeros(shape=(50, 5))
        self.coords = []
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
        self.object = self.window.create_oval(event.x - 4, event.y + 4, event.x + 4, event.y - 4, outline = 'red', fill = 'red')
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
        # self.coords[self.i,:] = [event.x, event.y, temp_coords[0], temp_coords[1], temp_coords[2]]
        self.coords.append([event.x, event.y, temp_coords[0], temp_coords[1], temp_coords[2]])
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
        
        # self.coords = self.coords[0:self.i,:]
        self.coords = np.array(self.coords)
        print('calibration points set: ')
        # np.save('coords.npy', self.coords)
        print(self.coords)
        return self.coords

class ParticleTrackingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Particle Tracking-ish")
        self.geometry("1400x800")

        self.working_path = ''
        self.paths = []
        
        # File input section
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(pady=10, padx=10, fill="x")
        
        self.file_label = ctk.CTkLabel(self.file_frame, text="Select a Working Directory:")
        self.file_label.pack(side="left", padx=10)
        
        self.file_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file)
        self.file_button.pack(side="left", padx=10)

        # Dropdown menu
        dropdown_label = ctk.CTkLabel(self.file_frame, text="Select Units:")
        dropdown_label.pack(side = 'left', pady=5, padx=10)

        self.units_var = tk.StringVar()
        # self.units_var.trace("w", self.on_parameter_change)
        parameter_dropdown = ctk.CTkOptionMenu(
            self.file_frame,
            variable=self.units_var,
            values=["m", "cm", "mm", 'inches'],  
        )
        parameter_dropdown.pack(side = 'left', pady=10, padx=10)
    
        # Tab control for different settings
        self.tab_control = ctk.CTkTabview(self)
        self.tab_control.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Tab for camera settings
        self.camera_tab = self.tab_control.add('Calibration Settings')
        self.create_camera_settings_tab(self.camera_tab)

        # Tab for particle detection settings
        self.detect_tab = self.tab_control.add('Particle Detection Settings')
        self.create_detection_settings_tab(self.detect_tab)

        # Tab for tracking settings
        self.track_tab = self.tab_control.add('Tracking Settings')
        self.create_tracking_settings_tab(self.track_tab)

        # Tab for Plots
        self.plots_tab = self.tab_control.add('Plot Results')
        self.create_plots_tab(self.plots_tab)

        # Buttons section
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10, padx=10, fill="x")
        
        self.calibrate_button = ctk.CTkButton(self.button_frame, text='Calibrate Cameras', command=self.calibrate_cameras)
        self.calibrate_button.pack(side="left", padx=10, pady=5)

        self.detect_button = ctk.CTkButton(self.button_frame, text='Detect Particles', command=self.detect_particles)
        self.detect_button.pack(side='left', padx=10, pady=5)
        
        self.track_button = ctk.CTkButton(self.button_frame, text='Track Particles', command=self.start_tracking)
        self.track_button.pack(side="left", padx=10, pady=5)
        
        self.clear_button = ctk.CTkButton(self.button_frame, text="Clear Console", command=self.clear_console)
        self.clear_button.pack(side="left", padx=10, pady=5)

        # self.console_toggle_button = ctk.CTkButton(self.button_frame, text="Pop-out Console", command=self.toggle_console)
        # self.console_toggle_button.pack(side="left", padx=10)
        
        # Console section
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.console_label = ctk.CTkLabel(self.console_frame, text="Console:")
        self.console_label.pack(anchor="nw", padx=10, pady=10)
        
        self.console_text = scrolledtext.ScrolledText(self.console_frame, state='disabled', height=10, bg='black', fg='white')
        self.console_text.pack(pady=5, padx=10, fill="both", expand=True)

        # Console window
        self.console_window = ConsoleWindow(self)
        
        # Redirect stdout and stderr to the console text widget
        sys.stdout = RedirectText(self.console_text)
        sys.stderr = RedirectText(self.console_text)
        
        self.console_in_main_window = True
    
# Tab creation Functions

    def create_camera_settings_tab(self, tab):
        self.camera_number_entry = None
        
        def on_camera_number_submit():
            try:
                self.num_cameras = int(self.camera_number_entry.get())
                self.cal_coords = [None] * self.num_cameras
                self.int_cal_folders = [None] * self.num_cameras
                self.images_folders = [None] * self.num_cameras
            except ValueError:
                print("Invalid number of cameras. Please enter an integer.")
                return
            
            for widget in camera_settings_frame.winfo_children():
                widget.destroy()

            for i in range(self.num_cameras):
                camera_frame = ctk.CTkFrame(camera_settings_frame)
                camera_frame.pack(pady=10, padx=10, fill="x")
                
                camera_label = ctk.CTkLabel(camera_frame, text=f"Camera {i+1}:")
                camera_label.pack(anchor="w", padx=5, pady=5)

                file_button = ctk.CTkButton(camera_frame, text="Select Extrinsic Calibration Image", command=lambda i=i: self.select_cal_points(i))
                file_button.pack(side="left", padx=5, pady= 5)
                
                folder_button = ctk.CTkButton(camera_frame, text="Select Folder of Images for Intrinsic Calibration", command=lambda i=i: self.select_int_cal_folder(i))
                folder_button.pack(side="left", padx=5, pady= 5)
                
                pic_file_button = ctk.CTkButton(camera_frame, text="Select Images Folder:", command=lambda i=i: self.browse_pics_folder(i))
                pic_file_button.pack(side="left", padx=5, pady=5)
        
        camera_number_frame = ctk.CTkFrame(tab)
        camera_number_frame.pack(pady=10, padx=10, fill="x")
        
        camera_number_label = ctk.CTkLabel(camera_number_frame, text="Number of Cameras:")
        camera_number_label.pack(side="left", padx=10)
        
        self.camera_number_entry = ctk.CTkEntry(camera_number_frame, width=50)
        self.camera_number_entry.pack(side="left", padx=10)
        
        camera_number_button = ctk.CTkButton(camera_number_frame, text="Submit", command=on_camera_number_submit)
        camera_number_button.pack(side="left", padx=10)

        save_coords_button = ctk.CTkButton(camera_number_frame, text='Export Calibration Coordinates to CSV', command=self.save_to_csv)
        save_coords_button.pack(side="left", padx=10)

        print_coords_button = ctk.CTkButton(camera_number_frame, text="Print Calibration Coordinates", command=self.print_coords)
        print_coords_button.pack(side="left", padx=10)

        load_coords_button = ctk.CTkButton(camera_number_frame, text='Load Coordinates from CSV', command= self.load_from_csv)
        load_coords_button.pack(side='left', padx= 10)

        all_cams_folder_button = ctk.CTkButton(camera_number_frame, text='Intrinsic cal folder', command=self.select_all_int_cal_folder)
        all_cams_folder_button.pack(side='left', padx= 10)

        # Scrollable frame for camera settings
        canvas = tk.Canvas(tab, background='grey17')
        scroll_y = tk.Scrollbar(tab, orient="vertical", command=canvas.yview)

        camera_settings_frame = ctk.CTkFrame(canvas)
        
        camera_settings_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=camera_settings_frame, anchor="nw")

        canvas.configure(yscrollcommand=scroll_y.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

    def create_tracking_settings_tab(self,tab):

        # Dropdown menu

        self.trackFrame = ctk.CTkFrame(tab)

        dropdown_label = ctk.CTkLabel(self.trackFrame, text="Select Tracking Method:")
        dropdown_label.pack(side = 'top', pady=5, padx=10)

        self.parameter_var = tk.StringVar()
        self.parameter_var.trace("w", self.on_parameter_change)
        parameter_dropdown = ctk.CTkOptionMenu(
            self.trackFrame,
            variable=self.parameter_var,
            values=["Nearest Neighbour", "Polynomial Extrapolation", "Wiener Filter", 'Lagrangian Coherent Tracking'],  
        )
        parameter_dropdown.pack(side = 'top', pady=10, padx=10)

        mintracklen_label = ctk.CTkLabel(self.trackFrame, text="Minimum Track Length:")
        mintracklen_label.pack(side = 'top', pady=5, padx=5)

        self.mintracklen_entry = ctk.CTkEntry(self.trackFrame, placeholder_text="Enter value")
        self.mintracklen_entry.pack(side= 'top', pady=5, padx=5)

        self.trackFrame.pack(side= 'left', padx= '10')

        # Text inputs
        param1_label = ctk.CTkLabel(tab, text="Track Initialization Search Radius:")
        param1_label.pack(side = 'top', pady=5, padx=1)

        self.param1_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")
        self.param1_entry.pack(side = 'top', pady=5, padx=1)

        param2_label = ctk.CTkLabel(tab, text="Tracking Search Radius:")
        param2_label.pack(side = 'top', pady=5, padx=1)

        self.param2_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")
        self.param2_entry.pack(side= 'top', pady=5, padx=1)

        param3_label = ctk.CTkLabel(tab, text="Frame Rate:")
        param3_label.pack(side = 'top', pady=5, padx=1)

        self.param3_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")
        self.param3_entry.pack(side= 'top', pady=5, padx=1)

        # Additional parameters
        self.additional_param_label = ctk.CTkLabel(tab, text="Polynomial Order:")
        self.additional_param_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")

        self.additional_param2_label = ctk.CTkLabel(tab, text="Noise Varience:")
        self.additional_param2_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")

        self.additional_param3_label = ctk.CTkLabel(tab, text="Size of Search Region for Coherant Particles:")
        self.additional_param3_entry = ctk.CTkEntry(tab, placeholder_text="Enter value")

        self.additional_param4_label = ctk.CTkLabel(self.trackFrame, text="Track Joining Search Radius:")
        self.additional_param4_entry = ctk.CTkEntry(self.trackFrame, placeholder_text="Enter value")

        # Toggle Switch
        self.switch_var = tk.StringVar(value='False')
        self.switch_var.trace("w", self.on_join_track_toggle)
        self.switch = ctk.CTkSwitch(self.trackFrame, text='Join Tracks', variable=self.switch_var, onvalue='True', offvalue='False')
        self.switch.pack(side= 'top', pady=5, padx=1)

        # Submit button
        submit_button = ctk.CTkButton(tab, text="Submit", command=self.submit_track_parameters)
        submit_button.pack(side= 'bottom',pady=5, padx=10)

        load_particles_button = ctk.CTkButton(tab, text='Load Particle Coordinates from file', command=self.load_particles)
        load_particles_button.pack(side= 'bottom', pady=5, padx=10)

    def create_detection_settings_tab(self, tab):

        # self.pic_folder_path = self.working_path + '/Images'

        # # File input section
        # self.pic_file_label = ctk.CTkLabel(tab, text="Select Images Folder:")
        # self.pic_file_label.pack(side="left", padx=10)

        # self.pic_file_entry = ctk.CTkEntry(tab, width=400, placeholder_text=self.pic_folder_path)
        # self.pic_file_entry.pack(side="left", padx=10)
        
        # self.pic_file_button = ctk.CTkButton(tab, text="Browse", command=self.browse_pics_folder)
        # self.pic_file_button.pack(side="left", padx=10)

        self.open_cal_window_button = ctk.CTkButton(tab, text= 'Calibrate Particle Detection Parameters', command= self.create_detection_param_tuning_window)
        self.open_cal_window_button.pack(side= 'left', padx= 10, pady=10)

        # Text inputs
        circ_frame = ctk.CTkFrame(tab)
        circ_frame.pack(side= 'top', pady=10, padx=10)

        circ_label = ctk.CTkLabel(circ_frame, text="Circularity:")
        circ_label.pack(side = 'left', pady=5, padx=5)

        self.min_circ_entry = ctk.CTkEntry(circ_frame, placeholder_text="Min value")
        self.min_circ_entry.pack(side = 'left', pady=5, padx=1) 

        self.max_circ_entry = ctk.CTkEntry(circ_frame, placeholder_text="Max value")
        self.max_circ_entry.pack(side = 'left', pady=5, padx=1)

        threshold_frame = ctk.CTkFrame(tab)
        threshold_frame.pack(side= 'top',pady=10, padx=10)

        threshold_label = ctk.CTkLabel(threshold_frame, text="Threshold:")
        threshold_label.pack(side = 'left', pady=5, padx=5)

        self.min_threshold_entry = ctk.CTkEntry(threshold_frame, placeholder_text="Min value")
        self.min_threshold_entry.pack(side = 'left', pady=5, padx=1) 

        self.max_threshold_entry = ctk.CTkEntry(threshold_frame, placeholder_text="Max value")
        self.max_threshold_entry.pack(side = 'left', pady=5, padx=1)

        area_frame = ctk.CTkFrame(tab)
        area_frame.pack(side= 'top',pady=10, padx=10)

        area_label = ctk.CTkLabel(area_frame, text="Area:")
        area_label.pack(side = 'left', pady=5, padx=5)

        self.min_area_entry = ctk.CTkEntry(area_frame, placeholder_text="Min value")
        self.min_area_entry.pack(side = 'left', pady=5, padx=1) 

        self.max_area_entry = ctk.CTkEntry(area_frame, placeholder_text="Max value")
        self.max_area_entry.pack(side = 'left', pady=5, padx=1)

        # Submit button

        eps_label = ctk.CTkLabel(tab, text='Epsilon:')
        eps_label.pack(side = 'left', pady=5, padx=5)

        self.detect_eps_entry = ctk.CTkEntry(tab, placeholder_text="Enter Value")
        self.detect_eps_entry.pack(side = 'left', pady=5, padx=1) 

        submit_button = ctk.CTkButton(tab, text="Submit", command=self.submit_detect_parameters)
        submit_button.pack(side= 'bottom',pady=5, padx=10)

        # Load data from CSV button
        # load_button = ctk.CTkButton(tab, text='Load Particle Positions from CSV', command=)
        # load_button.pack(side='left', padx= 5, pady= 5)

    def create_detection_param_tuning_window(self):

        # Callback functions for the trackbars to update the parameters
        def update_minThreshold(val):
            detector_params.minThreshold = val
            update_blobs()

        def update_maxThreshold(val):
            detector_params.maxThreshold = val
            update_blobs()

        def update_minArea(val):
            detector_params.filterByArea = True
            detector_params.minArea = val+1
            update_blobs()

        def update_maxArea(val):
            detector_params.filterByArea = True
            detector_params.maxArea = val
            update_blobs()

        def update_minCircularity(val):
            detector_params.filterByCircularity = True
            detector_params.minCircularity = val / 100
            update_blobs()

        # Function to update the image with detected blobs
        def update_blobs():
            detector = cv2.SimpleBlobDetector_create(detector_params)
            keypoints = detector.detect(image_inv)

            # Draw detected blobs as red circles
            img_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow('Blob Detector', img_with_keypoints)

        # Load image
        image_path = filedialog.askopenfilename()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_inv = cv2.bitwise_not(image)

        # Create a window
        cv2.namedWindow('Blob Detector', cv2.WINDOW_NORMAL)

        # Create SimpleBlobDetector parameters
        detector_params = cv2.SimpleBlobDetector_Params()

        # Create trackbars for the parameters
        cv2.createTrackbar('Min Threshold', 'Blob Detector', 10, 255, update_minThreshold)
        cv2.createTrackbar('Max Threshold', 'Blob Detector', 200, 255, update_maxThreshold)
        cv2.createTrackbar('Min Area', 'Blob Detector', 50, 100, update_minArea)
        cv2.createTrackbar('Max Area', 'Blob Detector', 75, 1000, update_maxArea)
        cv2.createTrackbar('Min Circularity', 'Blob Detector', 10, 100, update_minCircularity)

        # Initial update to display the image
        update_blobs()

        # Wait until a key is pressed
        cv2.waitKey(0)
        self.detector_params = (detector_params.minThreshold, detector_params.maxThreshold,
                              detector_params.minArea, detector_params.maxArea,
                              detector_params.minCircularity, detector_params.maxCircularity)
        print(self.detector_params)
        cv2.destroyAllWindows()

    def create_plots_tab(self, tab):

        self.plotTracks_button = ctk.CTkButton(tab, command= self.plotParticles, text= 'Plot Tracks')
        self.plotTracks_button.pack(side='left', padx= 10, pady= 10)

        self.plotCams_button = ctk.CTkButton(tab, command= self.plotCams, text= 'Plot Cameras')
        self.plotCams_button.pack(side= 'left', padx= 10, pady= 10)
        
# Command Functions

    def on_parameter_change(self, *args):
        algo = self.parameter_var.get()
        if algo == "Polynomial Extrapolation": 
            self.additional_param_label.pack(pady=5, padx=10)
            self.additional_param_entry.pack(pady=5, padx=10)
            self.additional_param2_label.pack_forget()
            self.additional_param2_entry.pack_forget()
            self.additional_param3_label.pack_forget()
            self.additional_param3_entry.pack_forget()
        elif algo =='Wiener Filter':
            self.additional_param2_label.pack(pady=5, padx=10)
            self.additional_param2_entry.pack(pady=5, padx=10)
            self.additional_param_label.pack_forget()
            self.additional_param_entry.pack_forget()
            self.additional_param3_label.pack_forget()
            self.additional_param3_entry.pack_forget()
        elif algo == 'Lagrangian Coherent Tracking':
            self.additional_param_label.pack(pady=5, padx=10)
            self.additional_param_entry.pack(pady=5, padx=10)
            self.additional_param3_label.pack(pady=5, padx=10)
            self.additional_param3_entry.pack(pady=5, padx=10)
            self.additional_param2_label.pack_forget()
            self.additional_param2_entry.pack_forget()
        else:
            self.additional_param_label.pack_forget()
            self.additional_param_entry.pack_forget()
            self.additional_param2_label.pack_forget()
            self.additional_param2_entry.pack_forget()
            self.additional_param3_label.pack_forget()
            self.additional_param3_entry.pack_forget()

    def on_join_track_toggle(self, *args):
        switch = self.switch_var.get()
        if switch == 'True':
            self.additional_param4_label.pack(pady=5, padx=10)
            self.additional_param4_entry.pack(pady=5, padx=10)
        else:
            self.additional_param4_label.pack_forget()
            self.additional_param4_entry.pack_forget()

    def submit_track_parameters(self):
        # Logic to handle the parameters
        algos = {'Nearest Neighbor': 'NN', 'Polynomial Extrapolation': 'Poly', 'Wiener Filter': 'Wiener', 'Lagrangian Coherant Tracking': 'LCS'}
        self.algo = algos[self.parameter_var.get()]

        self.NNepsilon = float(self.param1_entry.get())
        self.epsilon = float(self.param2_entry.get())
        self.dt = 1/float(self.param3_entry.get())
        self.minTrackLen = int(self.mintracklen_entry.get()) 
        self.noise = 0
        self.region = 0
        self.polyOrder = 0
        self.polyOrder = 0

        if self.algo == 'Wiener':
            self.noise = float(self.additional_param2_entry.get())
        elif self.algo == 'LCS':    
            self.region = float(self.additional_param3_entry.get())
            self.polyOrder = str(self.additional_param_entry.get())
        elif  self.algo == 'Poly':
            self.polyOrder = str(self.additional_param_entry.get())

        self.join_tracks = self.switch_var.get()
        if self.join_tracks == True:
            self.join_tracks_epsilon = float(self.additional_param4_entry.get())

        
                
        if  self.algo == 'Poly' or 'LCS':
            self.algo += self.polyOrder
        elif self.polyOrder == '1':
            self.algo = 'LinTerp'

        print('='*60, '\n','Algothrim:', self.algo,
                '\n Initialization threshold distance:', self.NNepsilon,
                '\n Tracking theshold distance:', self.epsilon,
                '\n Frame rate:', 1/self.dt, '(deltaT: ' + str(self.dt) + ')',
                '\n Join tracks:', self.join_tracks)
            
    def submit_detect_parameters(self):
        # Logic to handle the parameters
        self.detector_params = tuple(np.float32((self.min_threshold_entry.get(), self.max_threshold_entry.get(),
                              self.min_area_entry.get(), self.max_area_entry.get(),
                              self.min_circ_entry.get(), self.max_circ_entry.get())))
        self.detect_eps = float(self.detect_eps_entry.get())
        if len(self.detector_params) == 6: print('Detector Parameters Submitted!')
        # print('='*60, '\n','Algothrim:', algo,
        #         '\n Initialization threshold distance:', NNepsilon,
        #         '\n Tracking theshold distance:', epsilon,
        #         '\n Frame rate:', 1/dt, '(deltaT: ' + str(dt) + ')',
        #         '\n Join tracks:', join_tracks)

    def select_file(self, camera_index):
        file_path = filedialog.askopenfilename()
        print(f"Camera {camera_index + 1} selected file: {file_path}")

    def select_int_cal_folder(self, i):
        folder_path = filedialog.askdirectory()
        print(f"Camera {i + 1} selected folder: {folder_path}")
        self.int_cal_folders[i] = folder_path

    def select_all_int_cal_folder(self):
        folder_path = filedialog.askdirectory()
        self.int_cal_folders = [folder_path] * self.num_cameras
        print('Selected:', folder_path, 'for all cameras')

    def select_cal_points(self, i):
        print('='*60, '\n', 'Camera', i+1)
        self.root = CalPointSelection(self)
        self.cal_coords[i] = self.root.coords

    def save_to_csv(self):
        if self.working_path == '': 
            print('Please select a working directory first')
        else:    
            for i, arr in enumerate(self.cal_coords):
                df = pd.DataFrame(arr)
                df.to_csv(self.working_path + '/Cam' + str(i), header= ['Img X', 'Img Y', 'World X', 'World Y', 'World Z'], index= False)
            print('Calibration Coordinates saved to working directory')   

    def load_from_csv(self):
        paths = filedialog.askopenfilenames()
        self.cal_coords = []
        for path in paths:
            # self.cal_coords.append(pd.read_csv(path, delimiter=',', header=None, skiprows=1))
            self.cal_coords.append(np.genfromtxt(path, delimiter=',')[1:,:])
        if self.cal_coords != []:    
            print('Data loaded successfully')
        else:
            print('Import failed verify CSV file are not empty')

    def load_particles(self):
        path = filedialog.askopenfilenames()

        if path[0][-3:] == 'npy' and len(path) == 1:
            self.particles = np.load(path[0], allow_pickle=True)
            print('Data loaded :)')

        elif path[0][-3:] == 'txt' and len(path) > 1:
            files = sorted(path)
            pb = ProgressBar(len(files), 'Getting Data')
            for i, file in enumerate(path):
                if file[-3:] != 'txt': raise ValueError('Hmm the', str(i) + 'th file was not a .txt')
                df = pd.read_csv(file, sep='\s+', comment='#', header=None)
                xyz_df = df.iloc[:, 0:3]
                xyz_array = xyz_df.to_numpy()
                self.particles.append(xyz_array[:100,:])
                pb.Print(i+1)
        else:
            print('Only .npy Numpy arrys or .txt CSV files supported, note that if using CSV the data must be stored in 1 file per timestep')
            
    def print_coords(self):
        for j, arr in enumerate(self.cal_coords):
            print('Cam' + str(j), '\n', arr)

    def browse_file(self):
        self.working_path = filedialog.askdirectory()
        print(f"Selected file: {self.working_path}")

    def browse_pics_folder(self,i):
        folder = filedialog.askdirectory()
        self.images_folders[i] = folder
        print('Selected file for Camera',i,':',folder)

    def plotParticles(self):
        track_plot(self.particles, self.tracks, plot_density=0.5, plotParticles= False)
        plt.show()

    def plotCams(self):
        cal_plot(self.cameras, particles= self.particles, t=0)   
        plt.show()

# Functions for Primary Buttons

    def calibrate_cameras(self):
        self.units = str(self.units_var.get())
        if self.units == 'inches': print('Only sensible units supported'); # self.destroy()
        self.cameras = [calibrate(coords, self.int_cal_folders[j], self.units) for j, coords in enumerate(self.cal_coords)]
        print('Camera Calibration Complete')
        self.particles = [GetParticles(self.cameras, [cam.coords[:,:2] for cam in self.cameras], epsilon=1.5)]
        print(self.particles)

    def detect_particles(self):

        self.cam_pixel_coords = [DetectParticles(path, self.detector_params) for path in self.images_folders]
        # L-> [[p(t0) ..... p(tn)]_cam0 ..... [p(t0) ..... p(tn)]_camk] w/ [] = list & p(t) is the Nx2 np.array of particle positions


        # GetParticles wants as input[p_cam0 ... p_camk]_tn and returns p_tn
        print(self.detect_eps)
        self.particles = [GetParticles(self.cameras, [cam[t] for cam in self.cam_pixel_coords], epsilon = self.detect_eps) for t in range(len(self.cam_pixel_coords[0]))]
        # We want self.particles to be : [p_t0 ... p_tn]
        
        self.particles = [frame for frame in self.particles if not np.any(frame == None)]
    
    def start_tracking(self):
        
        self.tracks = GetTracksFBF(self.particles, algo=self.algo, epsilon= self.epsilon, NNepsilon= self.NNepsilon, deltaT= self.dt,
                                   region= self.region, Noise= self.noise, MinTrackLength= self.minTrackLen)
        
        if self.join_tracks == 'True':
            self.tracks = JoinTracks(self.tracks, epsilon=self.join_tracks_epsilon, deltaT=self.dt)
        
    def clear_console(self):
        self.console_text.config(state='normal')
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state='disabled')
        
    def toggle_console(self):
        if self.console_in_main_window:
            # Move console to the separate window
            self.console_text.pack_forget()
            self.console_text = scrolledtext.ScrolledText(self.console_window, state='disabled', height=10, bg="black", fg="white")
            self.console_text.pack(pady=5, padx=10, fill="both", expand=True)
            sys.stdout.output = self.console_text
            sys.stderr.output = self.console_text
            self.console_window.deiconify()
            self.console_toggle_button.configure(text="Embed Console")
        else:
            # Move console back to the main window
            self.console_window.console_text.pack_forget()
            self.console_text = scrolledtext.ScrolledText(self.console_frame, state='disabled', height=10, bg="black", fg="white")
            self.console_text.pack(pady=5, padx=10, fill="both", expand=True)
            sys.stdout.output = self.console_text
            sys.stderr.output = self.console_text
            self.console_window.withdraw()
            self.console_toggle_button.configure(text="Pop-out Console")
        
        self.console_in_main_window = not self.console_in_main_window

def Run():
    app = ParticleTrackingApp()
    app.mainloop()

Run()