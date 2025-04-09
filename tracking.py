from calibration import*
from plot_utils import*
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree



def DetectParticles(folder_path: str,
                 detector_params: tuple
) -> list[np.ndarray]:

    images = []
    img_folders = os.listdir(folder_path)
    img_folders = [file for file in img_folders if file != '.DS_Store']
    # for filename in img_folders: 
    #     if filename == '.DS_Store':
    #         img_folders.remove(filename)

    for filename in sorted(img_folders, key=lambda x: int(x[-9:-4])):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)

    coords_list = []
    
    for image in images:
        image_inv = cv2.bitwise_not(image)

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold, params.maxThreshold, params.minArea, params.maxArea, params.minCircularity, params.maxCircularity = detector_params
        
        detector = cv2.SimpleBlobDetector_create(params)

        particle_img_coords = detector.detect(image_inv)

        im_w_points = cv2.drawKeypoints(image, particle_img_coords, np.array([]), (0,0,225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        particle_img_coords = np.array(cv2.KeyPoint.convert(particle_img_coords))

        if particle_img_coords.size > 0:
            coords_list.append(particle_img_coords)
        else:
            coords_list.append(np.array([[None,None]]))

        # Uncommet this if you want to view the detected particles
        # cv2.imshow("particle_img_coords", im_w_points)
        # cv2.waitKey(0)   
        # print(len(particle_img_coords))


    return coords_list

def get_vec(point1: np.ndarray,
            point2: np.ndarray
) -> np.ndarray: 
    # Takes 2 points as inputs in hte form of 2 row vectors and returns a unit vector pointing from point1 to point2

    vec = (point2 - point1)/np.linalg.norm(point2 - point1, axis= 1)[:, np.newaxis]
    return vec

def GetParticles(cameras: list[object],
                  coords: list[np.ndarray],
                  vol_dims: tuple = None,
                  epsilon: float = 0.7
) -> tuple[np.ndarray]:
    # Input a list of camera objects and a corresponding list of pixel coordinates, the rays from the given pixels are calculated and their intersections 
    # are found, the dimensions of the interogation volume that limits the search area for the intersections 
    # epsilon is the threshold distance for intersections
    # multiple intersections correspond to a single particles due to there being more than 2 cameras and therefore intersection within some tolerance (epsilon) 
    # are clustered together

    all_vecs = []
    all_origins = []


    # Iterates over all cameras and gets gets the origin (camera position) and direction vector to 
    # define a ray corresponing to each detected particle                                  
    for i, camera in enumerate(cameras):
        points = camera.get_xyz(coords[i]).T
        # print('point cam',i,points)
        if np.all(points == None): continue
        origin = camera.translation_vec.T
        # print('tvec cam',i,origin)
        vecs = get_vec(origin, points).flatten()
        # print(vecs)
        origins = np.tile(origin, coords[i].shape[0]).flatten()
        all_vecs.append(vecs)
        all_origins.append(origins)

    if all_vecs == [] or all_origins == []: return np.array([[None,None, None]])

    vecs = np.hstack(all_vecs)
    origins = np.hstack(all_origins)

    imax = vecs.shape[0]
    intersects = []

    # Iterate over all the rays found computing the closest approachs between them and if they are 
    # under the threshold then compute the location of closest approachs 
    for i in range(0,imax,3):
        for j in range(0,imax,3):

            if np.cross(vecs[i: i+3],vecs[j: j+3]).all() == 0: break

            dist = np.linalg.norm(np.dot(origins[j: j+3] - origins[i: i+3],np.cross(vecs[i: i+3],vecs[j: j+3])/np.linalg.norm((np.cross(vecs[i: i+3],vecs[j: j+3])))))
            # print(origins[j: j+3])
            # print(origins[i: i+3])

            # print(dist)

            if dist < epsilon:                

                A = np.dot(vecs[i: i+3],vecs[j: j+3]); B = -np.dot(vecs[j: j+3],vecs[j: j+3]); C = -np.dot(vecs[i: i+3],vecs[i: i+3])
                D = np.dot(origins[j: j+3],vecs[j: j+3]) - np.dot(origins[i: i+3],vecs[j: j+3]); E = np.dot(origins[i: i+3],vecs[i: i+3]) - np.dot(origins[j: j+3],vecs[i: i+3])

                mat1 = np.array([[A, B],[C, A]]); mat2 = np.array([D, E])
                params = np.abs(np.linalg.inv(mat1) @ mat2)
                # print(params)
                if params.any() <= 0: break # change this to ensure that only particles in the measurement voume are taken

                point1 = origins[i: i+3] + params[0]*vecs[i: i+3]
                point2 = origins[j: j+3] + params[1]*vecs[j: j+3]
                coord = (point1 + point2)/2
                intersects.append(coord)

    if vol_dims is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = vol_dims
        intersects = [interx for interx in intersects if 
                    (interx[0] >= x_min) & (interx[0] <= x_max) &
                    (interx[1] >= y_min) & (interx[1] <= y_max) &
                    (interx[2] >= z_min) & (interx[2] <= z_max)]


    print(len(intersects), 'intersections found \n')
    
    particles = []

    nb_interesects = int((len(cameras)**2 - len(cameras))/2)

    # Cluster the intersection to get particles locations using a max search radius as well as the expected number of intersections per particle

    while intersects:
        intersect = intersects[0]
        distances = np.linalg.norm(intersects - intersect, axis=1)
        filtered = [(index, value) for index, value in enumerate(distances) if value < 5*epsilon] # The search zone to find clustered intresctions is hardcoded as 5*epsilon
        closest_points_indices = [index for index, _ in sorted(filtered, key=lambda x: x[1])[:nb_interesects]]

        closest_points = np.asarray(intersects)[closest_points_indices]
        if closest_points.shape[0] == nb_interesects:
            particles.append(np.mean(closest_points, axis=0))

        intersects = [x for idx, x in enumerate(intersects) if idx not in closest_points_indices]

    print(len(particles), 'particles found')

    if len(particles) == 0: return np.array([[None,None,None]])

    particles = np.array(particles)

    return particles


def NN(point: np.ndarray,
       coords: np.ndarray,
       epsilon: float = 1,
       i: int = None,
) -> tuple[np.ndarray, int]:
    
    # Returns the particle and its index in coords that is closest to point if it is within the maximum distnace of epsilon

    distances = np.linalg.norm(coords - point, axis= 1)

    point_index = np.argmin(distances)

    closest_point = coords[point_index,:]

    # print('dist', distances[point_index])

    if distances[point_index] > epsilon or point_index == i:
        return None, None
    else:
        return closest_point, point_index
    
def KDNN(point: np.ndarray,
         coords: np.ndarray,
         epsilon: float = 1,
         n: int = 1
)->np.ndarray:
    
    # Returns the particle and its index in coords that is closest to point if it is within the maximum distnace of epsilon

    tree = cKDTree(coords)
    distances, indices = tree.query(point, n)

    if  distances.size == 1 and distances < epsilon: 
        return coords[indices], indices
    elif n == 1: return None, None

    closest_points = [(coords[index], index) for i, index in enumerate(indices) if distances[i] < epsilon]

    if len(closest_points) == 0: return None, None

    closest_points, indices = zip(*closest_points)

    return np.array(closest_points), np.array(indices)

def LinTerp(points: np.ndarray,
            coords: np.ndarray,
            epsilon: float = 1
) -> tuple[np.ndarray, int]:
    
    # INPUTS:
    # points: Vertically stacked array of point coordinates
    # coords: Verically stacked array of candidate points
    # OUTPUTS:
    # (closest point coordinates, index of closest point)
    # Uses linear interpolation to estimate the postion of the particle at tn+1 from postions at tn and tn-1
    # and then finds the closest particle to the estimated postion within epsilon

    next_estimate = (2 * points[1,:] - points[0,:])[np.newaxis,:]

    return NN(next_estimate, coords, epsilon)

def PolyIterp(points: np.ndarray,
              coords: np.ndarray,
              deltaT: float,
              degree: int = 2,
              epsilon: float = 1,
              i: int = None,
) -> np.ndarray:
    
    # Similar to the linear intperlation but can do it with an arbitrary order polynomial
    # If you want a simple 1st order fit use the LinearInterp function for faster exectution

    if points.shape[0] < degree + 1: raise ValueError('Not enough points to fit a polynomial of given degree')
    
    t = np.arange(0,points.shape[0]+1) * deltaT
    t_learn = t[:-1]
    t_next  = t[-1]

    xfunc = np.polyfit(t_learn, points[:,0], degree)
    yfunc = np.polyfit(t_learn, points[:,1], degree)
    zfunc = np.polyfit(t_learn, points[:,2], degree)

    # print(points - np.array([np.polyval(xfunc, t_learn), np.polyval(yfunc, t_learn), np.polyval(zfunc, t_learn)]))

    next_estimate = np.array([np.polyval(xfunc, t_next), np.polyval(yfunc, t_next), np.polyval(zfunc, t_next)])
    # print('next esti', next_estimate)
    if coords.any() == None: return next_estimate
    # return NN(next_estimate, coords, epsilon, i)
    a, b = NN(next_estimate, coords, epsilon, i)
    return a, b, next_estimate[np.newaxis,:]

def Wiener(points: np.ndarray,
           coords: np.ndarray,
           epsilon: float,
           noise_variance: np.ndarray = np.array([0.05]),
) -> np.ndarray:
    
    n = points.shape[0]

    if  noise_variance.shape == (3,3):
        noise_variance == np.diagonal(noise_variance)


    signal_var = np.var(points, axis=0)
    H = signal_var / (signal_var + noise_variance)

    filtered_data = np.zeros_like(points)
    filtered_data[0] = points[0]
    
    for t in range(1, n):
        filtered_data[t] = H * points[t,:] + (1 - H) * filtered_data[t-1,:]
    
    predicted_position = H * points[-1,:] + (1 - H) * filtered_data[-1,:]

    a, b = NN(predicted_position, coords, epsilon)

    return a, b, predicted_position[np.newaxis, :]

def LSANN(coords1: np.ndarray,
         coords2: np.ndarray,
         epsilon_mat: np.ndarray,
)-> np.ndarray:
    
    # Matches the points in coords1 to the points in coords2 such that the sum of the
    # distances beween the matched points is minmised
    # epsilon_mat is of the shape as the cost (distnace) matrix and allows
    # for differesnth distance thesholds to be used for different tracks (eg for finding 2nd particel in track)

    # if coords2.shape[0] > coords1.shape[0]:
    #     coords1, coords2 = coords2, coords1
    #     epsilon_mat = epsilon_mat.T
    #     swap = True
    # else:
    #     swap = False

    distances = np.zeros((coords1.shape[0],coords2.shape[0]))
    decision_mat = np.zeros_like(distances)
    # temp = np.zeros_like(distances)

    for i, point1 in enumerate(coords1):
        for j, point2 in enumerate(coords2):
            distances[i,j] = np.linalg.norm(point2 - point1)

    row_indices, col_indices = linear_sum_assignment(distances)
    # print('dsit \n',np.round(distances, 1))
        
    indices = [(row_indices[i], col_indices[i]) for i in range(row_indices.shape[0])]
    indices = sorted(indices, key=lambda x: x[1])
    # print(indices)

    for i, j in indices:
        if distances[i,j] < epsilon_mat[i,j]: decision_mat[i,j] = 1
        # temp[i,j] = 1

    # print('raw mat \n', temp)

    # if swap is True: decision_mat = decision_mat.T

    return decision_mat

def GetTracksFBF(particles_in: list[np.ndarray],
                 epsilon: float = 1,
                 NNepsilon: float= 5,
                 deltaT: float = 1,
                 algo: str = 'NN',
                 MinTrackLength: int = 4,
                 region: float = 10,
                 nbParticles: int = 5,
                 Noise: np.ndarray = np.array([0.05]),
) -> list[np.ndarray]:
    
    # Make a deep copy of the input particles to avoid modifying the original data
    particles = particles_in.copy()

    # Initialize tracks with the first frame's particles
    tracks = [np.hstack((np.array([[int(0)]]), points[np.newaxis, :])) for points in particles[0]]

    pb = ProgressBar(len(particles_in), 'Getting Tracks')

    est = []

    # Iterate over each frame
    for i in range(1, len(particles)):
        frame = particles[i]

        # Iterate over each existing track
        for j, track in enumerate(tracks):

            # print(frame)

            if frame.size == 0 or np.any(frame == None):
                break
            elif track[-1,0] != i-1:
                continue

            if algo == 'NN':
                    next_particle, index = NN(track[-1,1:], frame, epsilon)

            elif algo == 'LinTerp':
                if track.shape[0] < 2:
                    next_particle, index = NN(track[-1,1:], frame, epsilon)
                else:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)

            elif algo[0:4] == 'Poly':
                n = int(algo[-1])

                if track.shape[0]<2:
                    next_particle, index = NN(track[-1,1:], frame, NNepsilon)
                elif track.shape[0]<n+1:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                else:
                    next_particle, index, next_estimate = PolyIterp(track[-n-1:,1:], frame, deltaT, n, epsilon) #changed nb of particles used for interp
                    est.append(next_estimate)
                    # print('next part',next_particle)

            elif algo == 'Wiener':

                if track.shape[0]<2:
                    next_particle, index = NN(track[-1,1:], frame, epsilon)
                elif track.shape[0]<4:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                else:
                    next_particle, index, _ = Wiener(track[:,1:], frame, epsilon, Noise)

            elif algo == 'Wiener2':

                if track.shape[0]<2:
                    next_particle, index = NN(track[-1,1:], frame, epsilon)
                elif track.shape[0]<10:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                else:
                    next_particle, index, _ = Wiener(track[:,1:], frame, epsilon, 9)


            elif algo[0:3] == 'LCS':
                n = int(algo[-1])

                if track.shape[0]<2:
                    next_particle, index = NN(track[-1,1:], frame, epsilon)
                elif track.shape[0]<n+1:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                else:
                    next_particle, index = LCS(track, tracks, frame,
                                                ThresholdFTLE = 1**-2, polydeg=n, deltaT=deltaT, epsilon=epsilon, region=region, nbParticles=nbParticles)
                    # print('nxt par', next_particle)

            elif algo[0:4] == '2LCS':
                n = int(algo[-1])

                if track.shape[0]<2:
                    next_particle, index = NN(track[-1,1:], frame, epsilon)
                elif track.shape[0]<n+1:
                    next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                else:
                    next_particle, index= LCSv2(track[-n-1:,:], tracks, frame,
                                                ThresholdFTLE = 1**-2, polydeg=n, deltaT=deltaT, epsilon=epsilon, region=region, nbParticles=nbParticles)
                    
            else: raise ValueError('Please Select a Valid Tracking Scheme')

            if next_particle is not None and index is not None:
                # Append the found particle to the current track

                track = np.vstack((track, np.hstack((np.array([int(i)]), next_particle))))

                # Update the track
                tracks[j] = track

                # Remove the found particle from the current frame
                frame = np.delete(frame, index, axis=0)

        pb.Print(i+1)

        # Create new tracks for the remaining particles in the current frame
        for point in frame:
            tracks.append(np.hstack((np.array([[int(i)]]), point[np.newaxis, :])))

    # est = np.vstack(est)
    # fog = plt.figure()
    # ax = plt.axes(projection = '3d')
    # ax.set_box_aspect([1.0, 1.0, 1.0])
    # ax.scatter(est[:,0],est[:,1],est[:,2], alpha=0.5, s=10, marker='x')
    # for track in tracks:
    #         if len(track.shape) == 1: track = track[np.newaxis,:]
    #         ax.plot3D(track[:,1],track[:,2],track[:,3], lw=2)
    #         ax.scatter(track[:,1],track[:,2],track[:,3], alpha=0.5, s=10)

    new_tracks = [track for track in tracks if track.shape[0] > (MinTrackLength - 1)]

    return new_tracks

def GetTracksFBFHun(particles_in: list[np.ndarray],
                epsilon: float = 1,
                NNepsilon: float= 5,
                deltaT: float = 1,
                algo: str = 'Poly2',
                fitLen: int = None,
                MinTrackLength: int = 4,
                region: float = 10,
                nbParticles: int = 5,
                Noise: float = np.array([0.05]),
) -> list[np.ndarray]:
    
    #  This funciton build the tracks by first getting next estimated positions of all tracks then assigning 
    # them using linear sum assignment to paritcles in the next frame.

    # Make a deep copy of the input particles to avoid modifying the original data
    particles = particles_in.copy()

    # Initialize tracks with the first frame's particles
    working_tracks = [np.hstack((np.array([[int(0)]]), point[np.newaxis, :])) for point in particles[0]]
    tracks = []
    lost_tracks = []
    est = []


    
    if algo[0] == 'P' or algo[0] == 'L' and fitLen == None: n=int(algo[-1]); fitLen = n + 1
    elif algo[0] == 'W': fitLen = int(algo[-2:]); n = fitLen-1
    elif algo == 'NN': pass
    else: n=int(algo[-1]); 
    
    pb1 = ProgressBar(len(particles) - 2, 'Getting Tracks')

    # Iterate over each frame
    for k, frame in enumerate(particles[1:]):
        pb1.Print(k)
        # print(i)

        # frame = particles[k]
        epsilon_mat = np.ones((len(working_tracks), frame.shape[0])) * epsilon
        if frame.size == 0: continue

        # print('frame',frame.shape)
        # print('tracks len', len(working_tracks))
        # print('working_tracks', working_tracks)
        
        estimates = []

        # Iterate over each existing track
        for j, track in enumerate(working_tracks):
            # print('track',track.shape)

            #  Get the next estimate for the track depending on track length and chosen predictor
            if track.shape[0] < 2 or algo == 'NN':
                next_estimate = track[-1,1:][np.newaxis,:]
                epsilon_mat[j,:] = NNepsilon

            elif track.shape[0] < fitLen or algo == 'Poly1':
                if track.shape[0] > n + 1:
                    _, _, next_estimate = PolyIterp(track[-n-1:,1:], frame, deltaT, n, epsilon)
                else:
                    next_estimate = (2 * track[-1,1:] - track[-2,1:])[np.newaxis,:]
            else:
                if algo[:4] == 'Poly':
                    _, _, next_estimate = PolyIterp(track[-fitLen:,1:], frame, deltaT, n, epsilon) 

                    est.append(next_estimate)
                elif algo[:3] == 'LCS':
                    _, _, next_estimate = LCS(track[-fitLen:,:], tracks, frame, 0, n, deltaT, epsilon)
                    # print('ttracks',track[-2:,1:],'\n est', next_estimate)
                    est.append(next_estimate)
                elif algo[:6] == 'Wiener':
                    _, _, next_estimate = Wiener(track[:,1:], frame, epsilon, fitLen, Noise)
                    est.append(next_estimate)
                else: raise ValueError('Please Select a Valid Tracking Scheme')

            estimates.append(next_estimate)

        # print('est', [ests.shape for ests in estimates])
        estimates = np.array(estimates)
        

        decision_mat = LSANN(estimates, frame, epsilon_mat)
        # print('\n' ,k, '\n')
        # print(decision_mat.shape)
        # print(decision_mat)

        new_tracks = []

        # Uses the desicion matrix to assign particles to working tracks, assign unpaired particles in the next frame to new tracks
        # and unpaired tracks to lost tracks
        for j, point in enumerate(frame):
            if np.all(decision_mat[:,j] == 0):
                new_tracks.append(np.hstack((np.array([int(k)]), point)))
            else:
                idx = np.nonzero(decision_mat[:,j])[0][0]
                # print('\n',decision_mat[:,j],idx)
                # if working_tracks[idx].shape[0] > 1:
                #     v0 = working_tracks[idx][-1,1:] - working_tracks[idx][-2,1:]
                #     v1 = point - working_tracks[idx][-1,1:]
                #     a = np.arccos(np.dot(v0,v1)/(np.linalg.norm(v1)*np.linalg.norm(v0)))
                #     if a < np.pi/3:
                #         working_tracks[idx] = np.vstack((working_tracks[idx], np.hstack((np.array([int(k)]), point))))
                # else:
                working_tracks[idx] = np.vstack((working_tracks[idx], np.hstack((np.array([int(k)]), point)))) 

        for i, track in enumerate(working_tracks):
            if np.all(decision_mat[i,:] == 0):
                lost_tracks.append(working_tracks.pop(i))

        for new_track in new_tracks:
            working_tracks.append(new_track[np.newaxis,:])

            # print('length of traks',[len(track) for track in working_tracks])
            # print('nb working tracks', len(working_tracks))
            # print('nb lost tracks', len(lost_tracks))

    for track in working_tracks:
        if track.shape[0] >= MinTrackLength:
            tracks.append(track)

    tracks = [track for track in (working_tracks + lost_tracks) if track.shape[0] >= MinTrackLength]

    return tracks, est

def JoinTracks(tracks: list[np.ndarray],
                epsilon: float,
                deltaT: float = None,
                n = 2
) -> list[np.ndarray]:
    
    new_tracks = tracks.copy()
    # new_tracks = [track[:,1:] for track in tracks]

    
    no_merge_counter = 0
    cnter = 0

    pb = ProgressBar(len(tracks), 'Joining Tracks')

    # This loop runs until the innner loop over each track complete without any
    # tracks being joined
    while no_merge_counter != len(tracks):

        starts = np.ones((1, 4))
        ends = np.ones((1, 4))

        # Get an array of the last 3 points in a track to get a 2nd order Polynomial 
        # estimate and the first points in all tracck
        for i, track in enumerate(new_tracks):
            # track = track[:,1:]
            if track.size == 0:
                starts = np.vstack((starts, np.ones((1, 4)) * np.inf))
                ends = np.vstack((ends, np.ones((3, 4)) * np.inf)) # change
            elif track.shape[0] < 3:
                # print('this should never happen')
                starts = np.vstack((starts, track[0, :]))
                ends = np.vstack((ends, np.vstack((np.ones((2,4)),track[-1, :]))))
            else:
                starts = np.vstack((starts, track[0, :]))
                ends = np.vstack((ends, track[-3:, :])) #change 

        starts = starts[1:,:]
        ends = ends[1:,:]

        i = 0
        no_merge_counter = 0

        # print(ends.shape)
        # print(ends[0:3,:])
        
        while i < len(new_tracks):

            if np.all(ends[3*i:3*i+3, :] == np.inf) or new_tracks[i].size == 0: # change
                i += 1
                no_merge_counter += 1
                continue
            
            _, index, _ = PolyIterp(ends[3*i:3*i+3, 1:], starts[:,1:], deltaT, n, epsilon, i)

            if index is not None and index != i and len(new_tracks[index]) > 0:

                # Last condition is there bc sometimes for some reason the rows of starts and the elements
                # of new_tracks dont correspond to eachother and then index points to an empty list element
                # this obvs shouldnt be happening but idk whats wrong so this'll do for now

                # print(ends[i,:], starts[index,:])
                # print(new_tracks[index])
                
                # Checking whether the segment joining the tracks isnt too 
                # out of line with the rest, max angle hardcoded as pi/3 / 60 deg
                v0 = new_tracks[i][-1,1:] - new_tracks[i][-2,1:]
                v1 = new_tracks[index][0,1:] - new_tracks[i][-1,1:]
                a = np.arccos(np.dot(v0,v1)/(np.linalg.norm(v1)*np.linalg.norm(v0)))
                
                if a < np.pi/3:
                    new_tracks[i] = np.vstack((new_tracks[i], new_tracks[index]))
                    new_tracks[index] = np.array([])  # Mark track as merged by making it empty
                    cnter += 1
                else:
                    no_merge_counter += 1
            else:
                no_merge_counter += 1

            i += 1

        pb.Print(no_merge_counter)

    new_tracks = [track for track in new_tracks if track.size != 0]

    print('\n', len(tracks), 'tracks reduced to', len(new_tracks), 'tracks')

    return new_tracks

def nNN(point: np.ndarray,
        coords: np.ndarray,
        epsilon: float = 1,
        n: int = 100,
) -> int:
    
    distances = np.linalg.norm(coords - point, axis= 1)

    point_indices = np.argsort(distances)[:n+1]

    closest_points = [(coords[index], index) for index in point_indices if distances[index] < epsilon]

    if len(closest_points) == 0: return None, None

    closest_points, indices = zip(*closest_points)

    return closest_points, indices

def nNNtrack(point: np.ndarray,
        coords: np.ndarray,
        epsilon: float = 1,
        n: int = 100,
) -> int:
    
    distances = np.linalg.norm(coords - point, axis= 1)
    # print(distances)

    point_indicies = np.argsort(distances)[1:n+1]

    closest_points = [(coords[index], index) for index in point_indicies if distances[index] < epsilon]

    if len(closest_points) == 0: return None, None

    closest_points, indicies = zip(*closest_points)

    return closest_points, indicies

def Wiener2(points, coords, epsilon, n=5):

    next_estimate = np.zeros((1,3))
    
    for i in range(3):
        # Calculate the autocorrelation of the time series
        r = np.correlate(points[:,i], points[:,i], mode='full')
        print(r.shape)
        r = r[len(r)//2:]  # Keep only the non-negative lags

        # Create the autocorrelation toeplitz matrix
        R = scipy.linalg.toeplitz(r[:n])

        # Compute the cross-correlation vector
        d = r[1:n+1]

        # R /= len(points)
        # d /= len(points)
        
        # Compute the Wiener filter coefficients
        h = np.linalg.solve(R, d)
        
        # Apply the filter to the time series to predict the next value
        next_estimate[0,i] = np.dot(h, points[-n:,i]) * 1.5


    a, b = NN(next_estimate, coords, epsilon)
    
    return a, b, next_estimate

def Wiener3(points, order):

    next_estimate = np.zeros((1,3))
        
    # Initialize arrays to hold predicted values for x, y, and z
    next_estimate = np.zeros(3)

    n = points.shape[0]
    
    # Iterate over each coordinate (x, y, z)
    for i in range(3):
        # Extract the specific coordinate series
        coord_series = points[:, i]
        
        # Create the autocorrelation matrix
        R = np.zeros((order, order))
        for j in range(order):
            for k in range(order):
                R[j, k] = np.dot(coord_series[j:n-order+j], coord_series[k:n-order+k]) / (n - order)

        # Create the cross-correlation vector
        r = np.zeros(order)
        for j in range(order):
            r[j] = np.dot(coord_series[j:n-order+j], coord_series[order:]) / (n - order)

        # Calculate the Wiener filter coefficients
        filter_coeffs = np.linalg.solve(R, r)
        
        # Predict the next value for this coordinate
        next_estimate[i] = np.dot(filter_coeffs, coord_series[-order:])
    
    return next_estimate[np.newaxis, :]
    


def LCSother(points: np.ndarray,
        coords: np.ndarray,
        tracks: list[np.ndarray],
        epsilon: float = 1,
        deltaT: float = 1,
        region: float = 5,
        nb_particles: int = 5,
) -> np.ndarray:
    
    temp_tracks = np.array([track[-2,:] for track in tracks])
    print(temp_tracks)
    
    neighbour_points, _, indicies = nNN(points[-2,:],temp_tracks,region, nb_particles)

    if neighbour_points == None: print('No particles found to determine Coherant structures'); return None
     
    print(len(neighbour_points))
    print('nn index', indicies)
    print('nieghbors', neighbour_points)


    diff1 = points[-2,:] - np.array(neighbour_points)

    temp_tracks = np.array([tracks[index][-1,:] for index in indicies])
    print(temp_tracks)

    diff2 = points[-1,:] - temp_tracks

    print(diff1.shape)
    print(diff2.shape)


    for i, points in enumerate(neighbour_points):

        D1 = np.tile(diff1[i,:], (3,1))

        D2 = np.tile(diff2[i,:][np.newaxis,:].T, (1,3))

        D = 1/D1 * D2

        delta = D.T @ D

        eig = np.max(np.linalg.eigvals(delta))

        FTLE = 1/deltaT * np.log(np.sqrt(eig))

        # print(eig)
        print(FTLE)

    return

def GetFTLE(points: np.ndarray,
            tracks: list[np.ndarray],
            deltaT: float = 1,
            region: float = 5,
            nb_particles: int = 15,
) -> np.ndarray:
    print('index',points[-1,0])

    temp_tracks1 = np.array([track[-2,1:] for track in tracks if track[-1,0] == points[-1,0]])
    temp_tracks2 = np.array([track[-1,1:] for track in tracks if track[-1,0] == points[-1,0]])
    print('temp trcs', temp_tracks1)
    
    neighbour_points1, indicies = nNNtrack(points[-2,1:], temp_tracks1, region)

    if neighbour_points1 == None or temp_tracks1.any == None: print('No particles found to determine coherant structures'); return None

    print('nieghbor points', neighbour_points1)

    neighbour_points2 = np.array([temp_tracks2[index] for index in indicies])

     
    # print(len(neighbour_points))
    # print('nn index', indicies)
    # print('nieghbors', neighbour_points)

    dist1 = np.linalg.norm(points[-2,1:] - np.array(neighbour_points1), axis=1)

    # temp_tracks2 = np.array([tracks[index][-1,:1] for index in indicies])
    
    dist2 = np.linalg.norm(points[-1,1:] - neighbour_points2, axis=1)

    FTLE = 1/deltaT * np.log(np.sqrt(dist2/dist1))

    V = (neighbour_points2 - neighbour_points1)/deltaT 
    # print(V.shape)
    # print(V)
    # print(FTLE)

    return FTLE, V

def GetCoherantVel(points: np.ndarray,
                    tracks: list[np.ndarray],
                    ThresholdFTLE: float,
                    deltaT: float = 1,
                    region: float = 5,
) -> np.ndarray:

    temp_tracks1 = np.array([track[-2,1:] for track in tracks if track[-1,0] == points[-1,0] and track.shape[0] >= 2])

    temp_tracks2 = np.array([track[-1,1:] for track in tracks if track[-1,0] == points[-1,0] and track.shape[0] >= 2])

    v = (points[-1,1:] - points[-2,1:])/deltaT

    if len(temp_tracks1) == 0 or len(temp_tracks2) == 0:
        # print('No particles found to determine coherant structures')
        return v
    
    neighbour_points1, indicies = nNNtrack(points[-2,1:], temp_tracks1, region)

    if neighbour_points1 == None or temp_tracks1.any == None: 
        # print('No particles found to determine coherant structures')
        return v

    # print('nieghbor points', neighbour_points1)

    neighbour_points2 = np.array([temp_tracks2[index] for index in indicies])

    dist1 = np.linalg.norm(points[-2,1:] - np.array(neighbour_points1), axis=1)
    
    dist2 = np.linalg.norm(points[-1,1:] - neighbour_points2, axis=1)

    FTLE = 1/deltaT * np.log(np.sqrt(dist2/dist1))

    V = (neighbour_points2 - neighbour_points1)/deltaT

    CoherantVel = np.mean([V[index] for index, FTLE in enumerate(FTLE) if FTLE < ThresholdFTLE], axis= 0)
 
    # print(V.shape)
    # print(V)
    # print(FTLE)

    return CoherantVel

def GetFTLEfield(tracks: list[np.ndarray]
) -> list[np.ndarray]:
    pass

def LCS(track: np.ndarray,
        tracks: list[np.ndarray],
        coords: np.ndarray,
        ThresholdFTLE: float,
        polydeg: int = 3,
        deltaT: float = 1,
        epsilon: float = 1,
        region: float = 5,
        nbParticles: int = 5,
) -> np.ndarray:
    
    def pos_func(params, ts):
        nbparams = len(params)
        pos = []
        for t in ts:
            x = sum([a*(t**i) for i, a in enumerate(params[:nbparams//3])])
            y = sum([a*(t**i) for i, a in enumerate(params[nbparams//3:nbparams//3*2])])
            z = sum([a*(t**i) for i, a in enumerate(params[nbparams//3*2:])])
            pos.append(np.hstack((x, y, z)))
        return np.vstack(pos)
    
    def vel_func(params, ts):
        nbparams = len(params)
        vel = []
        for t in ts:
            x = sum([a*(i+1)*(t**i) for i, a in enumerate(params[1:nbparams//3])])
            y = sum([a*(i+1)*(t**i) for i, a in enumerate(params[nbparams//3 + 1:nbparams//3*2])])
            z = sum([a*(i+1)*(t**i) for i, a in enumerate(params[nbparams//3*2 + 1:])])
            vel.append(np.hstack((x, y, z)))
        return np.vstack(vel)
    
    def loss_func(params, t, pos, tracks):
        CoherantVel = GetCoherantVel(pos, tracks, ThresholdFTLE, deltaT, region)
        loss = (pos_func(params, t) - pos[:,1:]) + (vel_func(params, t) - CoherantVel)
        return loss.ravel()
    
    a0 = np.zeros(((polydeg+1)*3))
    
    pos = track[:polydeg+1,:]

    t = np.arange(polydeg+1) * deltaT

    sol = least_squares(loss_func, a0, args= (t, pos, tracks), max_nfev=100)

    # print('sol is',sol.x)
    next_estimate = pos_func(sol.x, [(polydeg+2)*deltaT])

    # print(pos_func(sol.x, t) - pos[:,1:])

    # print('next est', next_estimate, sol.nfev, sol.cost)
    # print(sol.x)

    a, b = NN(next_estimate, coords, epsilon)

    return a, b, next_estimate

def LCSv2(track: np.ndarray,
          tracks: list[np.ndarray],
          coords: np.ndarray,
          deltaT: float,
          ThresholdFTLE: float,
          polydeg: int,
          epsilon: float,
          region: float,
          nbParticles: int = 5,
) -> np.ndarray:
    
    if track.shape[0] < polydeg + 1: raise ValueError('Not enough points to fit a polynomial of given degree')
    
    t = np.arange(0,track.shape[0]+1) * deltaT
    t_learn = t[:-1]
    t_next  = t[-1]

    xfunc = np.polyfit(t_learn, track[:,1], polydeg)
    yfunc = np.polyfit(t_learn, track[:,2], polydeg)
    zfunc = np.polyfit(t_learn, track[:,3], polydeg)

    next_estimate = np.array([np.polyval(xfunc, t_next), np.polyval(yfunc, t_next), np.polyval(zfunc, t_next)])

    # print(track, next_estimate)
    # print('coords.shape',coords.shape)
    # print(NN(next_estimate, coords, epsilon))
    neighbors, indices = nNN(next_estimate, coords, epsilon, nbParticles)

    if neighbors == None: return None, None

    # print(neighbors)

    neighborsVel = (neighbors - track[-1,1:])/deltaT

    CoherantVel = GetCoherantVel(track, tracks, ThresholdFTLE, deltaT, region)

    diff = np.linalg.norm(CoherantVel-neighborsVel, axis=1)

    index = np.argmin(diff)

    return neighbors[index], indices[index]



    """
    Plot the FTLE field in 3D.
    
    :param positions: 2D numpy array of 3D particle positions.
    :param ftle_field: 1D numpy array of FTLE values.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize FTLE values for color mapping
    norm_ftle = (ftle_field - np.min(ftle_field)) / (np.max(ftle_field) - np.min(ftle_field))
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=norm_ftle, cmap='inferno', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FTLE Field')
    fig.colorbar(scatter, label='FTLE')
    plt.show()



    """
    Plot the FTLE field in 3D.
    
    :param positions: 2D numpy array of 3D particle positions.
    :param ftle_field: 1D numpy array of FTLE values.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize FTLE values for color mapping
    norm_ftle = (ftle_field - np.min(ftle_field)) / (np.max(ftle_field) - np.min(ftle_field))
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=norm_ftle, cmap='inferno', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FTLE Field')
    fig.colorbar(scatter, label='FTLE')
    plt.show()

def HuNN2(coords1: np.ndarray,
         coords2: np.ndarray,
         epsilon_mat: np.ndarray,
)-> np.ndarray:

    if coords2.shape[0] > coords1.shape[0]:
        coords1, coords2 = coords2, coords1
        epsilon_mat = epsilon_mat.T
        swap = True
    else:
        swap = False

    distances = np.zeros((coords1.shape[0],coords2.shape[0]))
    decision_mat = np.zeros_like(distances)
    # temp = np.zeros_like(distances)

    for i, point1 in enumerate(coords1):
        for j, point2 in enumerate(coords2):
            distances[i,j] = np.linalg.norm(point2 - point1)

    np.fill_diagonal(decision_mat, 1000)

    row_indices, col_indices = linear_sum_assignment(distances)
    # print('dsit \n',np.round(distances, 1))
        
    indices = [(row_indices[i], col_indices[i]) for i in range(row_indices.shape[0])]
    indices = sorted(indices, key=lambda x: x[1])
    # print(indices)

    for i, j in indices:
        if distances[i,j] < epsilon_mat[i,j]: decision_mat[i,j] = 1
        # temp[i,j] = 1

    # print('raw mat \n', temp)

    if swap is True: decision_mat = decision_mat.T

    return decision_mat







def JoinTracks2(tracks: list[np.ndarray],
                 epsilon: float,
                 deltaT: float
)-> list[np.ndarray]:
    
    starts = np.array([track[0,1:] for track in tracks])
    # ends = np.array([PolyIterp(track[-3:,1:], np.array([None]), deltaT, 2) for track in tracks])
    ends = np.array([track[-1,1:] for track in tracks])

    distances = np.zeros((ends.shape[0],starts.shape[0]))

    for i, start in enumerate(ends):
        for j, end in enumerate(starts):
            distances[i,j] = np.linalg.norm(end - start)

    row_indices, col_indices = linear_sum_assignment(distances)


    new_tracks = []
    for i, j in zip(row_indices, col_indices):
        if distances[i,j] < epsilon and i != j:
            new_tracks.append(np.vstack((tracks[i],tracks[j])))
        elif i != j:
            new_tracks.append(tracks[i])
            new_tracks.append(tracks[j])
        else:
            new_tracks.append(tracks[i])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    a = np.vstack((starts,ends))
    
    
    ax.scatter(a[:, 0], a[:, 1], a[:, 2])



    return new_tracks

def JoinTracks3(tracks_in: list[np.ndarray],
                 particles: list[np.ndarray],
                 epsilon: float,
                 deltaT: float
)-> list[np.ndarray]:
        
    tracks = tracks_in.copy()
   
    # all_starts = np.array([track[0,1:] for track in tracks])
    # all_ends = np.array([track[-1,1:] for track in tracks])

    starts = np.array([PolyIterp(np.flip(track[:3,1:]), np.array([None]), deltaT, 2) for track in tracks])
    ends = np.array([PolyIterp(track[-3:,1:], np.array([None]), deltaT, 2) for track in tracks])


    # for t in range(0,len(particles)):

        # starts = np.array([start[1:] for start in all_starts if start[0] < t + step and start[0] > t - step])  
        # ends = np.array([end[1:] for end in all_ends if end[0] < t + step and end[0] > t - step])            

    epsilon_mat = np.ones((ends.shape[0], starts.shape[0])) * epsilon

    decision_mat = HuNN2(ends,starts, epsilon_mat)

    np.fill_diagonal(decision_mat, 0)

    print(decision_mat)

    new_tracks = []
    indices = []
    pb = ProgressBar(len(tracks), 'Joining Tracks')

    for i in range(len(tracks)):
        if i not in indices:
            if np.all(decision_mat[i,:] == 0):
                new_tracks.append(tracks[i])
            else:
                idx = np.nonzero(decision_mat[i,:])[0][0]
                indices.append(idx)
                new_tracks.append(np.vstack((tracks[i], tracks[idx])))
        pb.Print(i+1)


    
    print('\n', len(tracks), 'tracks reduced to', len(new_tracks), 'tracks')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2])
    ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2])

    
    return new_tracks
    


def GetTestData(NumberOfPoints: int = 100,
                VolDim: float = 100,
                DeltaT: float = 1,
                NumberOfSteps: int = 100,
                Noise: float = 0.02,
                GhostParticles: int = 3,
                Shuffle: bool = True,
                RandomSeed: bool = True
) -> list[np.ndarray]:

    def UniformFlow(coords: np.ndarray,
                    U: float,
                    V: float,
                    W: float
    ) -> np.ndarray:

        Vx = np.ones((NumberOfPoints,1)) * U

        Vy = np.ones((NumberOfPoints,1)) * V

        Vz = np.ones((NumberOfPoints,1)) * W

        return np.hstack((Vx, Vy, Vz))

    def Dipole(coords: np.ndarray,
               position: tuple[float] = (50,50,50),
               strength: float = 100,
               angle: float = 0
    ) -> np.ndarray:

        Vx = -strength/(2*np.pi)*(((coords[:,0:1] - position[0])**2 + (coords[:,2:3] - position[1])**2)* np.cos(angle) - 2 * (coords[:,0:1] - position[0]) * ((coords[:,0:1] - position[0]) * np.cos(angle) + (coords[:,1:2] - position[1]) * np.sin(angle)))/((coords[:,0:1] - position[0])**2 + (coords[:,1:2] - position[1])**2)**2
        
        Vy = -strength/(2*np.pi)*(((coords[:,0:1] - position[0])**2 + (coords[:,2:3] - position[1])**2)* np.cos(angle) - 2 * (coords[:,1:2] - position[1]) * ((coords[:,0:1] - position[0]) * np.cos(angle) + (coords[:,1:2] - position[1]) * np.sin(angle)))/((coords[:,0:1] - position[0])**2 + (coords[:,1:2] - position[1])**2)**2

        Vz = -strength/(2*np.pi)*(((coords[:,0:1] - position[0])**2 + (coords[:,2:3] - position[1])**2)* np.cos(angle) - 2 * (coords[:,2:3] - position[2]) * ((coords[:,0:1] - position[0]) * np.cos(angle) + (coords[:,1:2] - position[1]) * np.sin(angle)))/((coords[:,0:1] - position[0])**2 + (coords[:,1:2] - position[1])**2)**2

        return np.hstack((Vx, Vy, Vz))
    
    def Vortex(coords: np.ndarray,
               position: tuple[float] = (50,50),
               strength: float = 500
    ) -> np.ndarray:
        
        Vx = strength/(2*np.pi) * -(coords[:,1:2] - position[1]) / ((coords[:,0:1] - position[0])**2 + (coords[:,1:2] - position[1])**2) 

        Vy = strength/(2*np.pi) *  (coords[:,0:1] - position[0]) / ((coords[:,0:1] - position[0])**2 + (coords[:,1:2] - position[1])**2) 

        Vz = np.zeros((NumberOfPoints,1))

        return np.hstack((Vx, Vy, Vz))
    
    coords = []

    if RandomSeed == True:
        xmin, ymin, zmin = -30, 0, -15
        xmax, ymax, zmax =   0, 100, 65

        # xmin, ymin, zmin = 0, 0, 0
        # xmax, ymax, zmax =  100, 100, 100

        min = np.array([xmin, ymin, zmin])
        max = np.array([xmax, ymax, zmax])

        coords.append(min + np.random.rand(NumberOfPoints,3) * (max - min))
    else:
        
        x, y, z = -5, 37, 0
        xdisp, ydisp, zdisp = 5, 5, 5

        pos = np.array([x, y, z])
        disp = np.array([xdisp, ydisp, zdisp])

        coords.append(pos + np.random.rand(NumberOfPoints,3) * disp)


    max_disp = 0
    mean_disp = []

    # def Vel(t, coords):
    #     return (UniformFlow(coords,0.5,0,0) + Vortex(coords,(15,25), strength=25) + Vortex(coords,(35,25), strength=-25) + Vortex(coords,(0,0), 25)) * t
    
    # coords  = RK45(Vel, 0, coords, 60, vectorized=True)

    for i in range(1,NumberOfSteps*5):

        
        V = (UniformFlow(coords[i - 1],0.5,0,0.1) + 
            Vortex(coords[i - 1],(15,25), strength=25) + 
            Vortex(coords[i - 1],(35,25), strength=-25) + 
            Dipole(coords[i - 1], (100,100,100)) + 
            Vortex(coords[i - 1], (60,60,0), strength=35) 
        )
        

        max = np.max(np.linalg.norm(V, axis= 0))*DeltaT
        mean_disp.append(np.mean(np.linalg.norm(V, axis= 0)*DeltaT))
        if max > max_disp: max_disp = max

        # new_coords = (coords[i - 1] + V * DeltaT) + np.random.normal(0, Noise, coords[i - 1].shape)

        new_coords = (coords[i - 1] + V * DeltaT/5) 

        if Shuffle is True: np.random.shuffle(new_coords)

        coords.append(new_coords)

    Noise = np.array([[0.00456, -0.0004295, 0.003911],[-0.0004295, 0.003683, -0.001867],[0.003911, -0.001867, 0.01852]])

    coords = coords[::5]

    for i, frame in enumerate(coords):
        coords[i] = coords[i] + np.random.multivariate_normal(np.array([0,0,0]), Noise, coords[i].shape[0])
        coords[i] = np.vstack((frame, np.array((np.min(frame[:,0]),np.min(frame[:,1]),np.min(frame[:,2]))) + np.random.rand(GhostParticles,3) * np.array((np.max(frame[:,0]),np.max(frame[:,1]),np.max(frame[:,2])))))


    x_min, x_max =  0, 100
    y_min, y_max =  0, 100
    z_min, z_max =  0, 100
    coords = [arr[(arr[:, 0] >= x_min) & (arr[:, 0] <= x_max) &
                    (arr[:, 1] >= y_min) & (arr[:, 1] <= y_max) &
                    (arr[:, 2] >= z_min) & (arr[:, 2] <= z_max)]
                for arr in coords]

    return coords, max_disp, np.mean(mean_disp)