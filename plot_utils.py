import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import sys


class cal_plot():
    def __init__(self,
                cams: list[object],
                standard_plot: bool = False,
                particles: list[np.ndarray] = None,
                t: int = None
    ) -> None:
        
        self.cams = cams
        self.standard_plot = standard_plot
        self.all_particles = particles
        self.t = t if t is not None else 0
        self.particles = particles[self.t] if particles is not None else None
        self.plot_rays_enabled = False
        self.plot_particles_enabled = True

        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d', )
        self.ax.set_box_aspect([1.0, 1.0, 1.0])

        self.seg_len = np.mean([np.linalg.norm(self.cams[i].translation_vec) for i in range(len(self.cams))])/5
        self.plot_initial_elements()

        if standard_plot:
            self.plot_cams()
            if self.particles is not None:
                self.plot_particles(self.particles)
        else:
            self.plot_cams()

        self.create_widgets()

    def plot_initial_elements(self):
        x = np.array([0, self.seg_len])
        self.ax.plot3D(x, 0, 0, 'red')
        self.ax.plot3D(0, x, 0, 'green')
        self.ax.plot3D(0, 0, x, 'blue')
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_zlabel('Z [mm]')

    def plot_cams(self):
        for cam in self.cams:
            xt = np.hstack((np.zeros((3,1)), cam.rot_mat @ np.array([[self.seg_len, 0, 0]]).T)) + cam.translation_vec
            yt = np.hstack((np.zeros((3,1)), cam.rot_mat @ np.array([[0, self.seg_len, 0]]).T)) + cam.translation_vec
            zt = np.hstack((np.zeros((3,1)), cam.rot_mat @ np.array([[0, 0, self.seg_len]]).T)) + cam.translation_vec

            self.ax.plot3D(xt[0,:], xt[1,:], xt[2,:], 'red')
            self.ax.plot3D(yt[0,:], yt[1,:], yt[2,:], 'green')
            self.ax.plot3D(zt[0,:], zt[1,:], zt[2,:], 'blue')
        set_axes_equal(self.ax)

    def plot_rays(self,):
        # This functoin dosnt work rn because it can only plot the rays from the extrinsic calibration, for it to work
        # at each timestep the pixel coordinates detected for each cam at each frame has to be passed to the cam.get_xyz methond
        for cam in self.cams:
            for point in cam.coords[:, :2]:
                segment = np.hstack((cam.translation_vec, cam.get_xyz(point[np.newaxis, :], 250)))
                self.ax.plot3D(segment[0], segment[1], segment[2])

    def plot_particles(self, particles):
        self.ax.scatter(particles[:,0], particles[:,1], particles[:,2])

    def update_t(self, val):
        self.t = int(val)
        self.particles = self.all_particles[self.t] if self.all_particles is not None else None
        self.redraw()

    def toggle_plot_rays(self, event):
        self.plot_rays_enabled = not self.plot_rays_enabled
        self.redraw()

    def toggle_plot_particles(self, event):
        self.plot_particles_enabled = not self.plot_particles_enabled
        self.redraw()

    def redraw(self):
        self.ax.cla()
        self.plot_initial_elements()
        self.plot_cams()
        if self.plot_particles_enabled and self.particles is not None:
            self.plot_particles(self.particles)
        if self.plot_rays_enabled:
            self.plot_rays()

    def create_widgets(self):
        axcolor = 'lightgoldenrodyellow'

        if self.all_particles is not None and len(self.all_particles) > 1:
            ax_t = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor=axcolor)
            self.s_t = Slider(ax_t, 't', 0, len(self.all_particles)-1, valinit=self.t, valstep=1)
            self.s_t.on_changed(self.update_t)

        ax_button_rays = plt.axes([0.8, 0.1, 0.15, 0.04])
        self.button_rays = Button(ax_button_rays, 'Toggle Rays', color=axcolor, hovercolor='0.975')
        self.button_rays.on_clicked(self.toggle_plot_rays)

        ax_button_particles = plt.axes([0.8, 0.025, 0.15, 0.04])
        self.button_particles = Button(ax_button_particles, 'Toggle Particles', color=axcolor, hovercolor='0.975')
        self.button_particles.on_clicked(self.toggle_plot_particles)

    def show(self):
        plt.show()

class track_plot():
    def __init__(self,
                 particles: list[np.ndarray],
                 tracks: list[np.ndarray],
                 plot_density: float = 1,
                 plotParticles: bool = True,
                 bounds: tuple = (0,0),
    ):
        self.t = len(particles) - 1 if len(particles) > 1 else 1
        self.particles = particles

        # # if tracks[0].shape[1] == 4:
        # #     self.tracks = [track[:,1:] for track in tracks]
        # else: self.tracks = tracks
        self.tracks = tracks

        self.plot_density = plot_density
        self.plotParticles = plotParticles
        self.bounds = bounds
        
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_box_aspect([1.0, 1.0, 1.0])

        self.seg_len = 15
        x = np.array([0, self.seg_len])

        self.ax.plot3D(x, 0, 0, 'red')
        self.ax.plot3D(0, x, 0, 'green')
        self.ax.plot3D(0, 0, x, 'blue')
       


        if bounds[0] != 0 and bounds[1] != 0:
            self.ax.set_xlim3d(left=bounds[0], right=bounds[1]) 
            self.ax.set_ylim3d(bottom=bounds[0], top=bounds[1]) 
            self.ax.set_zlim3d(bottom=bounds[0], top=bounds[1])

        self.plot_data()
        self.create_widgets()

    def plot_data(self):
        self.ax.cla()
        
        x = np.array([0, self.seg_len])
        self.ax.plot3D(x, 0, 0, 'red')
        self.ax.plot3D(0, x, 0, 'green')
        self.ax.plot3D(0, 0, x, 'blue')
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_zlabel('Z [mm]')
        
        if self.bounds[0] != 0 or self.bounds[1] != 0:
            self.ax.set_xlim3d(left=self.bounds[0], right=self.bounds[1]) 
            self.ax.set_ylim3d(bottom=self.bounds[0], top=self.bounds[1]) 
            # self.ax.set_zlim3d(bottom=self.bounds[0], top=self.bounds[1])
            self.ax.set_zlim3d(bottom=20, top=40)

        if self.plot_density <= 1:
            skip = int(1 / self.plot_density)
        else:
            skip = self.plot_density

        # self.ax.scatter(self.particles[0][:,0], self.particles[0][:,1], self.particles[0][:,2], alpha=0.5, s=20, marker='x', color='red')
        if self.plotParticles:
            for points in self.particles[1:self.t+2:skip]:
                self.ax.scatter(points[:,0], points[:,1], points[:,2], alpha=0.5, s=10)

        for track in self.tracks:
            if len(track.shape) == 1: 
                track = track[np.newaxis, :]
                print(track[0,0])
            if track[0,0] <= self.t:
                x = int(self.t - track[0,0])
                self.ax.plot3D(track[:x,1], track[:x,2], track[:x,3], lw=2)
                # if not self.plotParticles:
                #     self.ax.scatter(track[:x:skip,1], track[:x:skip,2], track[:x:skip,3], alpha=0.5, s=10)

    def update_plot_density(self, val):
        self.plot_density = val/100
        self.plot_data()
        plt.draw()

    def update_plot_time(self, val):
        self.t = val
        self.plot_data()
        plt.draw()

    def toggle_plot_particles(self, event):
        self.plotParticles = not self.plotParticles
        self.plot_data()
        plt.draw()

    def create_widgets(self):
        axcolor = 'lightgoldenrodyellow'
        
        if len(self.particles) > 1:
            ax_t = plt.axes([0.1, 0.04, 0.65, 0.03], facecolor=axcolor)
            self.s_t = Slider(ax_t, 't', 0, len(self.particles)-1, valinit=self.t, valstep=1)
            self.s_t.on_changed(self.update_plot_time)

        ax_density = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
        self.s_density = Slider(ax_density, 'Density', 0, 100, valinit=self.plot_density)
        self.s_density.on_changed(self.update_plot_density)

        ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(ax_button, 'Toggle Particles', color=axcolor, hovercolor='0.975')
        self.button.on_clicked(self.toggle_plot_particles)

    def show(self):
        plt.show()

class ProgressBar():
    def __init__(self, total, prefix):
        self.total = total
        self.prefix = prefix
        print(' ')

    def Print(self, i, barLength=25, fill = '█'):
        suffix = 'Complete'
        progress = i / self.total
        filledLength = int(barLength * progress)
        bar = fill * filledLength + '-' * (barLength - filledLength)
        percentage = int(progress * 100)
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percentage}% {suffix}')
        sys.stdout.flush()

class Vector():
    def __init__(self,
                 particles: list[np.ndarray],
                 tracks: list[np.ndarray],
                 plot_density: float = 1,
                 plotParticles: bool = True,
                 bounds: tuple = (0,0),
    ):
        self.t = len(particles) - 1 if len(particles) > 1 else 1
        self.particles = particles

        self.tracks = tracks

        self.plot_density = plot_density
        self.plotParticles = plotParticles
        self.bounds = bounds
        
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_box_aspect([1.0, 1.0, 1.0])

        self.seg_len = 15
        x = np.array([0, self.seg_len])

        self.ax.plot3D(x, 0, 0, 'red')
        self.ax.plot3D(0, x, 0, 'green')
        self.ax.plot3D(0, 0, x, 'blue')

        # Initialize lists to store the velocity vectors and positions
        X, Y, Z, U, V, W = [], [], [], [], [], []

        skip = 5

        for particle_data in self.tracks:
            for i in range(len(particle_data) - 1, skip):
                t1, x1, y1, z1 = particle_data[i]
                t2, x2, y2, z2 = particle_data[i+skip]
                x, y, z = particle_data[i+skip]/particle_data[i]
                
                # Compute velocity components
                dt = t2 - t1
                if dt != 0 and z < 30 and z>20:
                    u = (x2 - x1) / dt
                    v = (y2 - y1) / dt
                    w = (z2 - z1) / dt
                    
                # Store positions and velocity components
                X.append(x)
                Y.append(y)
                Z.append(z)
                U.append(u)
                V.append(v)
                W.append(w)

        # Convert to numpy arrays for plotting
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)

        # Plotting the 3D vector field
        self.ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        


def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


