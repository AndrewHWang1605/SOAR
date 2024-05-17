"""
MIT License

Copyright (c) 2024 Andrew Wang, Bryan Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Implement track
"""
import numpy as np


class Track:
    def __init__(self, track_config):
        self.track_config = track_config
        self.generateTrackRep(track_config)        

    # Convert curvilinear coordinates to global
    def CLtoGlobal(self, state):
        """
        curvilinear state: [s, ey, epsi, vx, vy, w, delta]
        global state: [x, y, theta, vx, vy, w, delta]
        """
        s, ey, epsi, vx_cl, vy_cl, w, delta = state
        track_x, track_y, track_psi = self.getTrackPosition(state[0])
        
        x = track_x + ey*np.sin(track_psi)
        y = track_y - ey*np.cos(track_psi)
        theta = (track_psi + epsi) % (2*np.pi)
        vx = vx_cl*np.cos(theta) + vy_cl*np.sin(theta)
        vy = vx_cl*np.sin(theta) - vy_cl*np.cos(theta)
        
        global_state = np.array([x, y, theta, vx, vy, w, delta])
        return global_state

    def getCurvature(self, s):
        nearest_s_ind = np.argmin(np.abs(self.s - s%self.total_len))
        return self.track_curvature[nearest_s_ind]

    def getTrackPosition(self, s):
        nearest_s_ind = np.argmin(np.abs(self.s - s%self.total_len))
        track_x, track_y, track_psi = self.track_xypsi[nearest_s_ind]
        return track_x, track_y, track_psi

    def generateTrackFromCurvature(self, segment_curvature, segment_change, ds):
        """
        Generates a piecewise-constant curvature track with discretization ds, using curvature information specified
        segment_curvature specifies the curvature of each piecewise segment
        segment_change denotes distance along the full track at which we switch pieces (last element equals total length of track)
        ds specifies track length discretization size
        """
        total_len = segment_change[-1]
        s = np.arange(0, total_len, ds)
        track_curvature = np.zeros(s.shape[0])
        track_xypsi = np.zeros((s.shape[0], 3))
        track_xypsi[0,:] = np.zeros(3)
        segment_counter = 0
        for i in range(s.shape[0]):
            if (s[i] > segment_change[segment_counter]):
                segment_counter += 1
            track_curvature[i] = segment_curvature[segment_counter]
            
            if (i < s.shape[0]-1):
                theta = ds*track_curvature[i]
                dtheta = track_xypsi[i,2] - 0.5*ds*track_curvature[i]
                dx = ds if track_curvature[i] == 0 else np.abs(2/track_curvature[i]*np.sin(theta/2))
                track_xypsi[i+1,:] = track_xypsi[i,:] + np.array([dx*np.cos(dtheta), dx*np.sin(dtheta), -theta])
                track_xypsi[i+1,2] %= (2*np.pi)
            else:
                theta = ds*track_curvature[i]
                dtheta = track_xypsi[i,2] - 0.5*ds*track_curvature[i]
                dx = ds if track_curvature[i] == 0 else np.abs(2/track_curvature[i]*np.sin(theta/2))
                loop_back_start = track_xypsi[i,:] + np.array([dx*np.cos(dtheta), dx*np.sin(dtheta), -theta])
                err = track_xypsi[0,:]-loop_back_start
                err[2] %= (2*np.pi)
                print("Track Closure Error", err)
                assert np.all(err < np.array([0.1, 0.1, 0.01]))

        return total_len, s, track_curvature, track_xypsi


class OvalTrack(Track):
    def __init__(self, track_config):
        super().__init__(track_config)

    def generateTrackRep(self, track_config):
        track_half_width = track_config["track_half_width"]
        straight_len = track_config["straight_length"]
        curve_rad = track_config["curve_radius"]

        segment_curvature = [0, -1/curve_rad, 0, -1/curve_rad]
        segment_change = np.cumsum([straight_len, curve_rad*np.pi, straight_len, curve_rad*np.pi])
        ds = 0.05

        self.total_len, self.s, self.track_curvature, self.track_xypsi = self.generateTrackFromCurvature(segment_curvature, segment_change, ds) 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    track_config = {"track_half_width":10, "straight_length":100, "curve_radius":90}
    ovaltrack = OvalTrack(track_config)
    
    """
    curvilinear state: [s, ey, epsi, vx, vy, w, delta]
    global state: [x, y, theta, vx, vy, w, delta]
    """
    cl_state = np.array([120, -7, np.pi/4, 25, 30, 0, 0])
    glob_state = ovaltrack.CLtoGlobal(cl_state)
    track_x,track_y,track_psi = ovaltrack.getTrackPosition(cl_state[0])
    print("Curvilinear State", cl_state)
    print("Global State", glob_state)

    xypsi = ovaltrack.track_xypsi
    plt.scatter(xypsi[:,0], xypsi[:,1], s=0.2, marker='*')
    plt.scatter(xypsi[:,0] - track_config["track_half_width"]*np.sin(xypsi[:,2]), xypsi[:,1] + track_config["track_half_width"]*np.cos(xypsi[:,2]), s=0.2, marker='o') # Left track limit
    plt.scatter(xypsi[:,0] + track_config["track_half_width"]*np.sin(xypsi[:,2]), xypsi[:,1] - track_config["track_half_width"]*np.cos(xypsi[:,2]), s=0.2, marker='o') # Right track limit
    
    plt.scatter(track_x, track_y)
    plt.quiver(glob_state[0], glob_state[1], np.cos(glob_state[2]), np.sin(glob_state[2]))
    plt.quiver(glob_state[0], glob_state[1], glob_state[3], glob_state[4], color='r')
    plt.legend(["Centerline", "Left Bound", "Right Bound", "Track pos", "Pose", "Velo"])
    plt.axis('equal')
    plt.show()


