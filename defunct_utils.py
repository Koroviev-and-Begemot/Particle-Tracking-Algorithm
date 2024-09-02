def construct_rays(cameras: list[object],
                   coords: list[np.ndarray],
                   vol_dim: float = 100
):
    i = 0
    range = np.linspace(-vol_dim,vol_dim,100)
    rays = [np.tile(range, (2,1))]
    print(rays[0].shape)
    for camera in cameras:
        points = camera.get_xyz(coords[i]).T
        print(points.shape)
        origin = camera.translation_vec
        origin.flatten()
        print(origin.shape)

        x = np.tile(range, (points.shape[0],1))
        y = np.tile(range, (1,points.shape[0]))
        coefs  = get_seg(origin, points)
        z = - (coefs[:,3][:,np.newaxis] + coefs[:,0][:,np.newaxis]*x + coefs[:,1][:,np.newaxis]*y)/coefs[:,2][:,np.newaxis]
        print(z.shape)
        rays.append(z)
    
    return rays

def get_seg(point1: np.ndarray,
            point2: np.ndarray
) -> np.ndarray:
    coefs = np.zeros((point2.shape[0],4))
    print(coefs.shape)
    coefs[:,0] =  2/(point2[:,0] - point1[0])
    coefs[:,1] = -1/(point2[:,1] - point1[1])
    coefs[:,2] = -1/(point2[:,2] - point1[2])
    coefs[:,3] = -coefs[:,0]*point1[0] - coefs[:,1]*point1[1] - coefs[:,2]*point1[2]
    return coefs

def get_intersects(cameras: list[object],
                  coords: list[np.ndarray],
                  vol_dim: float = 250
) -> list[np.ndarray]:
    # Input a list of camera objects and a corresponding list of pixel coordinates, the rays from the given pixels are calculated and their intersections 
    # are found, the dimensions of the interogation volume that limits the search area for the intersections

    vecs   = []
    points = []
    origin = []
    j = 0
    for camera in cameras:
        points.append(camera.get_xyz(coords[j]).T)
        origin.append(camera.translation_vec.T)
        vecs.append(get_vec(origin[j], points[j]))
        j += 1
    
    i = 0
    for point in origin:
        temp_origin = origin.copy(); temp_origin.remove(point)
        temp_vecs = vecs.copy();     temp_vecs.remove(vecs[i])
        for vec1 in vecs[i]:
            # ray_1 point = point, dir = vec1
            print(vec1)
            for temp_point in temp_origin:
                print(temp_point)
                for temp_vec in temp_vecs:
                    for vec2 in temp_vec:
                        # ray_2: point = temp_point, dir = vec2
                        a = np.linalg.norm(np.dot(point - temp_point,np.cross(vec1,vec2)/np.linalg.norm((np.cross(vec1,vec2)))))
                        
        i += 1

def NearestNeighbour(particles: list[np.ndarray],
                     epsilon: float = 25
) -> list[np.ndarray]:
    
    # Tracks each particle present at the 1st frame through everyframe by taking its nearest neighbor

    closest_points_indicies = []

    for i, frame in enumerate(particles[:-1]):

        closest_point_index = np.ones_like(frame[:,0])*np.nan

        candidates = particles[i+1].copy()

        for j, particle in enumerate(frame):

            distances = np.linalg.norm(candidates - particle, axis=1)

            filtered = [(index, value) for index, value in enumerate(distances) if value < epsilon]

            if filtered == []:
                closest_point_index[j] = np.nan
            else:
                closest_point_index[j] = int(sorted(filtered, key=lambda x: x[1])[0][0])
                candidates[int(closest_point_index[j]),:] = float('inf')

        closest_points_indicies.append(closest_point_index)

    return closest_points_indicies

def LinInterpNN(particles: list[np.ndarray],
                          deltaT: float = 1,
                          epsilon: float = 30,
) -> list[np.ndarray]:
    
    initial_tracks = GetTracks(particles[:2],epsilon=epsilon)
    tracks = []

    for init_track in initial_tracks:

        track = init_track

        for i, _ in enumerate(particles[2:], 1):
            
            if track.shape[0] < 2: pass

            else:

                next_estimate = (2 * track[i,:] - track[i-1,:])[np.newaxis,:]

                next_point = GetTracks([next_estimate, particles[i+1]],epsilon=epsilon)[0][-1,:]

                track = np.vstack((track,next_point))
            
        tracks.append(track)

    return tracks    

def tracklets(particles: list[np.ndarray],
              epsilon: float,
              n: int,
) -> list[np.ndarray]:
    
    tracks = []

    for i, _ in enumerate(particles[::n]):

        track = LinInterpNN(particles[i*n:i*n+n], epsilon=epsilon)
        tracks += track
    
    return tracks    

def GetTracks(particles: list[np.ndarray],
              TrackAlgo: str = 'NearestNeighbour',
              epsilon: float = 1
) -> list[np.ndarray]:
    
    # INPUTS:
    # particles: each list element is an array of particle coordinates corresponding to a timestep
    # TrackAlgo: Stracking scheme used to track particles
    # OUTPUTS:
    # tracks: each list element corresponds to the coordinates of a single particle position at each timestep
    
    tracks = []

    if TrackAlgo == 'NearestNeighbour': closest_points = NearestNeighbour(particles, epsilon)
    else: raise ValueError('Select a valid tracking scheme') 
    
    for old_index in range(particles[0].shape[0]):

        track = particles[0][old_index,:]

        for i, points in enumerate(particles[1:]):

            if m.isnan(closest_points[i][old_index]) == False: 

                point_index = int(closest_points[i][old_index])
                new_point = np.array([points[point_index,0],points[point_index,1],points[point_index,2]])

                track = np.vstack((track,new_point))
                
                old_index = point_index

            else: break

        # tracks.append(track[1:,:])
        if len(track.shape) == 1: track = track[np.newaxis,:]
        tracks.append(track)

    return tracks

def join_tracks(tracks, epsilon):
    """
    Joins tracks together if the end and beginning of two tracks are within some distance epsilon.

    Parameters:
    - tracks: List of arrays, where each array is a list of 3D coordinates representing a track.
    - NN: Function that takes a point, a list of coordinates, and epsilon, and returns a tuple of
          the closest particle in the list of coordinates to the point within distance epsilon, 
          and the index of that particle.
    - epsilon: Maximum distance to consider for joining tracks.

    Returns:
    - A new list of joined tracks.
    """
    # Create a copy of the tracks to modify
    joined_tracks = tracks.copy()

    new_tracks = []

    # Iterate over all tracks
    for i, track in enumerate(joined_tracks):
        if track is None:
            continue

        # Get the end point of the current track
        end_point = track[-1,:]

        # Check all other tracks to find the closest starting point
        for j, other_track in enumerate(joined_tracks):
            if i != j and other_track is not None:
                start_point = other_track[0,:]
                # Use NN function to find the closest starting point within epsilon
                nearest_point, idx = NN(end_point, start_point[np.newaxis,:], epsilon)
                if nearest_point is not None:
                    # If a track is found to be close enough, merge the tracks
                    merged_track = np.vstack((track, other_track))  
                    new_tracks.append(merged_track)
                    # Mark the tracks as None to avoid reusing them
                    joined_tracks[i] = None
                    joined_tracks[j] = None
                    merged = True
                    break

        # If no merging happened for this track, add it to new_tracks
        if joined_tracks[i] is not None:
            new_tracks.append(track)

    # Update joined_tracks with new_tracks
    joined_tracks = new_tracks

    return joined_tracks

def JoinTracks(tracks: list[np.ndarray],
               epsilon: float
) -> list[np.ndarray]:
    
    tracks = tracks.copy()
    new_tracks = tracks.copy()
    
    for i, inital_track in enumerate(tracks):

        point = inital_track[-1,:]

        candidates = np.array([coord[0,:] for coord in tracks])

        _, index = NN(point, candidates, epsilon)

        if index != None:

            new_track = np.vstack((inital_track, tracks[index]))

            new_tracks.append(new_track)

            new_tracks.remove(inital_track)
            new_tracks.pop(index)

    return new_tracks

def JoinTracks2(tracks: list[np.ndarray],
                epsilon: float
) -> list[np.ndarray]:

    old_tracks = tracks.copy()
    
    # for i in range(1):

    coords = []
    starts = np.ones((1,3))

    for i, track in enumerate(old_tracks):
        coords.append((i, track[0,:], track[-1,:]))
        starts = np.vstack((starts,track[0,:]))

    starts = starts[1:,:]

    candidates = coords.copy()

    new_tracks = []

    for track1 in candidates:

        merged_flag = False

        # if track1[0] == None:
        #     pass
        # else:
        #     for track2 in candidates:

        #         if track2[0] == None: 
        #             pass
        #         else:
        #             dist = np.linalg.norm(np.array(track1[2]) - np.array(track2[1]))

        #             if dist < epsilon and track1[0] != track2[0]:

        #                 coords[track1[0]] = None
        #                 coords[track2[0]] = None

        #                 new_tracks.append(np.vstack((old_tracks[track1[0]], old_tracks[track2[0]])))
        #                 merged_flag = True
        #                 break

        #     if merged_flag == False:   
        #         new_tracks.append(old_tracks[track1[0]])

        _, index = NN(track1[2], starts, epsilon)

        if index != None and track1[0] != index:

            new_tracks.append(np.vstack((old_tracks[track1[0]], old_tracks[index])))
            coords[track1[0]] = None
            coords[index]     = None
        elif track1[0] != index:
            new_tracks.append(old_tracks[track1[0]])
        
    old_tracks = new_tracks.copy()  

    return new_tracks

def JoinTracksV3(tracks: list[np.ndarray],
                 epsilon: float
) -> list[np.ndarray]:
    
    new_tracks = tracks.copy()

    no_merge_counter = 0
    
    while no_merge_counter < len(tracks):

        coords = []
        starts = np.ones((1,3))
        ends   = np.ones((1,3))

        for i, track in enumerate(new_tracks):
            if track.any() == None:
                starts = np.vstack((starts, np.ones((1,3))*np.inf))
                ends   = np.vstack((ends, np.ones((1,3))*np.inf))
            else:
                coords.append((i, track[0,:], track[-1,:]))
                starts = np.vstack((starts,track[0,:]))
                ends   = np.vstack((ends, track[-1,:]))

        starts = starts[1:,:]
        ends   = ends[1:,:]

        i = 0
        no_merge_counter = 0

        while i < len(tracks):

            if ends[i,:].any() == np.inf:
                i += 1
                break

            _, index = NN(ends[i,:], starts, epsilon, i)
            
            if index != None and index != i:
                new_tracks[i] = np.vstack((new_tracks[i], new_tracks[index]))
                new_tracks[index] = np.array((None))
            elif index == None:
                no_merge_counter += 1

            i += 1

    new_tracks = [track for track in new_tracks if track.any() != None]

    return new_tracks

# print('intersects: \n', intersects)

    # particles = []
    # for intersect in intersects:
    #     distances = []
    #     print('intesect', intersect)
    #     for test_intersect in intersects:

    #         dist = np.linalg.norm(intersect - test_intersect)
    #         distances.append(dist)

    #     print('distances: \n', distances)
    #     filtered = [(index, value) for index, value in enumerate(distances) if value < 5]
    #     # distances = [dist for dist in distances if dist < 10*epsilon]
    #     closest_points = [index for index, _ in sorted(filtered, key=lambda x: x[1])[:len(cameras)]]
    #     print('closest_points: \n', closest_points)
    #     print('points: \n', np.asarray(intersects)[closest_points])

    #     particles.append(np.mean(np.asarray(intersects)[closest_points], axis=0))
    #     print('mean: \n', particles[-1])

    #     intersects = [x for x in intersects if x not in np.asarray(intersects)[closest_points]]
    #     # intersects.pop(closest_points)
    #     print('intersects after pop: \n', intersects)

    # print(len(particles), 'particles found')
    # print(temp_inter)
    # print(particles)

def get_R_mat(psi,theta,phi):
    # World frame to camera frame rotaiton matrix
    return np.array([[  m.cos(psi) * m.cos(theta)                                       ,                       m.sin(psi) * m.cos(theta)                   ,       -m.sin(theta)       ],
                     [- m.sin(psi) * m.cos(phi) + m.cos(psi) * m.sin(theta) * m.cos(phi),   m.cos(psi) * m.cos(phi) + m.sin(psi) * m.sin(theta) * m.sin(phi),  m.cos(theta) * m.sin(phi)],
                     [  m.sin(psi) * m.sin(phi) + m.cos(psi) * m.sin(theta) * m.cos(phi),  -m.cos(psi) * m.sin(phi) + m.sin(psi) * m.sin(theta) * m.cos(phi),  m.cos(theta) * m.cos(phi)]])


def calibrate_algabraic(coords, img_path, pixel_size = 3.5/1280):

    p_x = coords[:,0]; p_y = coords[:,1]
    x   = coords[:,2]; y   = coords[:,3]; z = coords[:,4]


    img_shape =  cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape  
    
    def equations(vars):
        x_t, y_t, z_t, phi, theta, psi, f = vars

        x_p = ( p_x - img_shape[1]/2)*pixel_size/f
        y_p = (-p_y + img_shape[0]/2)*pixel_size/f

        zc = np.sqrt((x_t - x)**2 + (y_t - y)**2 + (z_t - z)**2)

        f1 = (( m.cos(psi) * m.cos(theta)) * x_p + 
              (-m.sin(psi) * m.cos(phi) + m.cos(psi) * m.sin(theta) * m.cos(phi)) * y_p + 
              ( m.sin(psi) * m.sin(phi) + m.cos(psi) * m.sin(theta) * m.cos(phi))) * zc + x_t - x
        
        f2 = (( m.sin(psi) * m.sin(theta)) * x_p + 
              ( m.cos(psi) * m.cos(phi) + m.sin(psi) * m.sin(theta) * m.sin(phi)) * y_p + 
              (-m.cos(psi) * m.sin(phi) + m.sin(psi) * m.sin(theta) * m.cos(phi))) * zc + y_t - y
        
        f3 = ((-m.sin(theta)) * x_p + 
              ( m.cos(theta) * m.sin(phi)) * y_p + 
              ( m.cos(theta) * m.cos(phi))) * zc + z_t - z

        return np.hstack([f1, f2, f3]) 
    
    result = opt.least_squares(
                            fun = equations, x0  = (0, 0, 0, np.pi/2, np.pi/4, np.pi/4, 5),
                            max_nfev= 5000, loss='soft_l1', 
                            bounds= ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 2.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 30])
                        )

    values = result.x
    values[3:-2] = (values[3:-2]*180/np.pi)%360    

    print(result.message)
    print('x_t, y_t, z_t = ', values[0:3])
    print('phi, theta, psi = ', values[3:-1])
    print('f = ', values[-1])
    print('Nb of iterations = ', result.nfev)
    return values


def calibrate_algabraic_2(
    coords: np.ndarray,
    img_path: str,
    sensor_size: float = 0.685,
    f: float = 0.4
) ->tuple[np.ndarray, np.ndarray, np.ndarray]: 

    imgx, imgy =  cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
    pixel_size = sensor_size/imgx

    def params_to_matrix(
        angles: np.ndarray,
        f: float,
    ) -> np.ndarray:
        
        sinphi, sintheta, sinpsi = np.sin(angles)
        cosphi, costheta, cospsi = np.cos(angles)

        A = np.array((
            (cospsi*costheta, cospsi*sintheta*cosphi - sinpsi*cosphi, cospsi*sintheta*cosphi + sinpsi*sinphi),
            (sinpsi*sintheta, sinpsi*sintheta*sinphi + cospsi*cosphi, sinpsi*sintheta*cosphi - cospsi*sinphi),
            (      -sintheta,        costheta*sinphi,                        costheta*cosphi),
        ))
        
        A[:, :2] /= f
        
        return A.T


    def equations(
        params: np.ndarray,
        f: float,
        xy_pf: np.ndarray,
        xyz: np.ndarray,
    ) -> np.ndarray:
        
        xyzt, angles = np.split(params, (3,))
        zc = np.linalg.norm(xyzt - xyz, axis=1)
        A = params_to_matrix(angles, f)

        # Homogeneous affine: rotation transform
        f123 = np.hstack((xy_pf, np.ones((zc.size, 1)),))@A * zc[:, np.newaxis] + xyzt - xyz
        return f123.ravel()


    def calibrate(
        coords: np.ndarray,
        f: float,
        pixel_size: float,
        imgx: int,
        imgy: int,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        
        p_xy, xyz = np.split(coords, (2,), axis=1)

        pel_transform = pixel_size * np.array((
            ( 1,  0),
            ( 0, -1),
            (-0.5*imgx, 0.5*imgy),
        ))
        xy_pf = np.hstack((p_xy, np.ones((p_xy.shape[0], 1)))) @ pel_transform

        result = opt.least_squares(
            fun=equations,
            args=(f, xy_pf, xyz),
            x0=(0, 0, 0, 0, 0, 0),
            max_nfev= 10**4
        )
        if not result.success:
            raise ValueError(result.message)

        xyzt, angles = np.split(result.x, (3,))
        A = params_to_matrix(angles, f)

        zc = np.linalg.norm(xyzt - xyz, axis=1, keepdims=True)
        print(xy_pf)
        X_ = A @ np.transpose(np.hstack((xy_pf, np.ones_like(zc),))*zc) + xyzt[:, np.newaxis]

        acc = np.mean(np.linalg.norm(X_ - np.transpose(xyz), axis = 0))

        print(result.message)
        print('x_t, y_t, z_t =', xyzt)
        print('phi, theta, psi =', angles)
        print('iterations =', result.nfev)
        print(result.cost)
        print('Mean error =', acc)
        print(X_)
        return xyzt, np.rad2deg(angles), A

    xyzt, angles, A = calibrate(coords, f , pixel_size, imgx, imgy)
    return xyzt, angles, A

def cal(
        coords: np.ndarray,
        f: float,
        path: str,
        sensor_size: float = 0.685,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    uv, xyz = np.split(coords, (2,), axis=1)

    imgx, imgy =  cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape
    pixel_size = sensor_size/imgx

    B = np.array([[   1   ,   0   ,      -imgx/2  ],
                  [   0   ,  -1   ,       imgy/2  ],
                  [   0   ,   0   ,   1/pixel_size]])
    
    xy_c = -B @ np.vstack((uv.T, np.ones_like(uv[:,0])))*pixel_size

    def params_to_matrix(   
        angles: np.ndarray,
        f: float,
    ) -> np.ndarray:
        
        sinphi, sintheta, sinpsi = np.sin(angles)
        cosphi, costheta, cospsi = np.cos(angles)

        A = np.array((
            (cospsi*costheta, cospsi*sintheta*cosphi - sinpsi*cosphi, cospsi*sintheta*cosphi + sinpsi*sinphi),
            (sinpsi*sintheta, sinpsi*sintheta*sinphi + cospsi*cosphi, sinpsi*sintheta*cosphi - cospsi*sinphi),
            (      -sintheta,        costheta*sinphi,                        costheta*cosphi),
        ))
        
        A[:2, :] /= f
        
        return A

    def equations(params: np.ndarray,
                  xy_c: np.ndarray,
                  xyz: np.ndarray,
                  f: float,
    ) -> np.ndarray:
        
        T, angles = np.split(params, (3,))

        zc = np.linalg.norm(xyz - T, axis= 1)

        A = params_to_matrix(angles, f)

        eqn = A @ xy_c * zc + np.asmatrix(T).T - np.asmatrix(xyz).T

        return np.asarray(eqn).ravel()
    
    result = opt.least_squares(
            fun=equations,
            args=(xy_c, xyz, f),
            x0=(0, 0, 0, 0, 0, 0),
            max_nfev= 10**4
        )
    if not result.success:
            raise ValueError(result.message)
    
    xyzt, angles = np.split(result.x, (3,))

    print(result.message)
    print('x_t, y_t, z_t =', xyzt)
    print('phi, theta, psi =', np.rad2deg(angles))
    print('iterations =', result.nfev)
    print(result.cost)

def GetTracksTBT(particles_in: list[np.ndarray],
                epsilon: float = 1,
                algo: str = 'NN',
                deltaT: float = 1,
                MinTrackLength: int = 3,
                region: float = 10,
                nbParticles: int = 5,
                Noise: float = 0.15
) -> list[np.ndarray]:
    
    # INPUTS:
    # particles_in: each list element is an array of particle coordinates corresponding to a timestep
    # algo: Tracking scheme used to track particles
    # epslion: see above
    # OUTPUTS:
    # tracks: each list element corresponds to the coordinates of a single particle position at each timestep

    particles = particles_in.copy()
    tracks = []

    for i, _ in enumerate(particles):

        for particle in particles[i]:

            track = particle[np.newaxis,:]
            
            for j, frame in enumerate(particles[i:]):  

                if frame.size == 0: break

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
                        next_particle, index = NN(track[-1,1:], frame, epsilon)
                    elif track.shape[0]<n+1:
                        next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                    else:
                        next_particle, index = PolyIterp(track[-n-1:,1:], frame, deltaT, n, epsilon)

                elif algo == 'Wiener':

                    if track.shape[0]<2:
                        next_particle, index = NN(track[-1,1:], frame, epsilon)
                    elif track.shape[0]<4:
                        next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                    else:
                        next_particle, index = Weiner(track[:,1:], frame, epsilon, Noise)

                elif algo == 'LCS':
                    n = int(algo[-1])

                    if track.shape[0]<2:
                        next_particle, index = NN(track[-1,1:], frame, epsilon)
                    elif track.shape[0]<n+1:
                        next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                    else:
                        next_particle, index = LCS(track, tracks, frame, ThresholdFTLE = 1**-2, polydeg=n, deltaT=deltaT, epsilon=epsilon, region=region, nbParticles=nbParticles)

                elif algo == 'LCSv2':
                    n = int(algo[-1])

                    if track.shape[0]<2:
                        next_particle, index = NN(track[-1,1:], frame, epsilon)
                    elif track.shape[0]<n+1:
                        next_particle, index = LinTerp(track[-2:,1:], frame, epsilon)
                    else:
                        next_particle, index = LCS(track, tracks, frame, ThresholdFTLE = 1**-2, polydeg=n, deltaT=deltaT, epsilon=epsilon, region=region, nbParticles=nbParticles)


                else: raise ValueError('Please Select a Valid Tracking Scheme')

                if index == None or particles[j+i].size == 0: break 

                elif index != None and next_particle.any() != None: 

                    track = np.vstack((track,next_particle))

                    particles[j+i] = np.delete(particles[j+i], index, axis=0)

            tracks.append(track)

    tracks = [track for track in tracks if len(track) > (MinTrackLength - 1)]

    return tracks


def GetTracksFBF(particles_in: list[np.ndarray],
                 


                epsilon: float = 1,
                deltaT: float = 1,
                algo: str = 'NN', 
) -> list[np.ndarray]:

        particles = particles_in.copy()

        tracks = [points[np.newaxis,:] for points in particles[0]]

        for i, _ in enumerate(particles):

        for j, frame in enumerate(particles[i:]):
            
            for k, track in enumerate(tracks): 

                if frame.size == 0: break

                if algo == 'NN':
                    next_particle, index = NN(track[-1,:], frame, epsilon)

                elif algo == 'LinTerp':
                    if track.shape[0] < 2:
                        next_particle, index = NN(track[-1,:], frame, epsilon)
                    else:
                        next_particle, index = LinTerp(track[-2:,:], frame, epsilon)

                elif algo[0:4] == 'Poly':
                    n = int(algo[-1])

                    if track.shape[0]<2:
                        next_particle, index = NN(track[-1,:], frame, epsilon)
                    elif track.shape[0]<n+1:
                        next_particle, index = LinTerp(track[-2:,:], frame, epsilon)
                    else:
                        next_particle, index = PolyIterp(track[-n-1:,:], frame, deltaT, n, epsilon)

                elif algo == 'Wiener':

                    if track.shape[0]<2:
                        next_particle, index = NN(track[-1,:], frame, epsilon)
                    elif track.shape[0]<4:
                        next_particle, index = LinTerp(track[-2:,:], frame, epsilon)
                    else:
                        next_particle, index = Weiner(track, frame, epsilon)

                else: raise ValueError('Please Select a Valid Tracking Scheme')

                if index == None or particles[j+i].size == 0: break 

                elif index != None and next_particle.any() != None: 

                    track = np.vstack((track,next_particle))

                    tracks[k] = track

                    particles[j+i] = np.delete(particles[j+i], index, axis=0)

            tracks =  tracks + [point[np.newaxis,:] for point in particles[j+i]]

        return tracks