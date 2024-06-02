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
import matplotlib.pyplot as plt
import casadi as ca

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
        track_x, track_y, track_psi = self.getTrackPosition(s)
                
        x = track_x - ey*np.sin(track_psi)
        y = track_y + ey*np.cos(track_psi)
        theta = (track_psi + epsi) % (2*np.pi)
        vx = vx_cl*np.cos(theta) + vy_cl*np.sin(theta)
        vy = vx_cl*np.sin(theta) - vy_cl*np.cos(theta)
        
        global_state = np.array([x, y, theta, vx, vy, w, delta])
        return global_state

    def getCurvature(self, s):
        s = self.normalizeS(s)
        curvature = np.interp(s, self.s, self.track_curvature)
        return curvature

    def getTrackPosition(self, s):
        s = self.normalizeS(s)
        xypsi = [0]*3
        for i in range(3):
            xypsi[i] = np.interp(s, self.s, self.track_xypsi[:,i])
        track_x, track_y, track_psi = xypsi
        return track_x, track_y, track_psi

    def normalizeS(self, s):
        """ Maps s to range [0, self.total_len] """
        return np.mod(np.mod(s, self.total_len) + self.total_len, self.total_len)
        
    def signedSDist(self, s1, s2):
        """ Returns signed s1-s2, but accounting for wraparound effects """
        s1 = self.normalizeS(s1)
        s2 = self.normalizeS(s2)
        if (s1-s2 > 0.5*self.total_len): # s1 way ahead of s2, so wrap around and measure as behind s2
            return (s1-self.total_len) - s2
        elif (s1-s2 < -0.5*self.total_len): # s1 way behind s2, so wrap around and measure as ahead s2
            return s1 - (s2-self.total_len)
        else: # s1, s2 in same half of track
            return s1-s2
    
    def getTrackLength(self):
        return self.total_len

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
                dtheta = track_xypsi[i,2] + 0.5*ds*track_curvature[i]
                dx = ds if track_curvature[i] == 0 else np.abs(2/track_curvature[i]*np.sin(theta/2))
                track_xypsi[i+1,:] = track_xypsi[i,:] + np.array([dx*np.cos(dtheta), dx*np.sin(dtheta), theta])
                track_xypsi[i+1,2] %= (2*np.pi)
            else:
                theta = ds*track_curvature[i]
                dtheta = track_xypsi[i,2] + 0.5*ds*track_curvature[i]
                dx = ds if track_curvature[i] == 0 else np.abs(2/track_curvature[i]*np.sin(theta/2))
                loop_back_start = track_xypsi[i,:] + np.array([dx*np.cos(dtheta), dx*np.sin(dtheta), theta])
                err = track_xypsi[0,:]-loop_back_start
                err[2] = 1-np.cos(err[2])
                print("Track Closure Error", err)
                assert np.all(err < np.array([0.1, 0.1, 1-np.cos(0.1)]))

        return total_len, s, track_curvature, track_xypsi

    def plotTrack(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(15,8))
        ax.plot(self.track_xypsi[:,0], self.track_xypsi[:,1], linestyle='--', color='lightgrey', zorder=-1)
        ax.plot(self.track_xypsi[:,0] - self.track_config["track_half_width"]*np.sin(self.track_xypsi[:,2]), self.track_xypsi[:,1] + self.track_config["track_half_width"]*np.cos(self.track_xypsi[:,2]), linestyle='-', color='grey', zorder=-1) # Left track limit
        ax.plot(self.track_xypsi[:,0] + self.track_config["track_half_width"]*np.sin(self.track_xypsi[:,2]), self.track_xypsi[:,1] - self.track_config["track_half_width"]*np.cos(self.track_xypsi[:,2]), linestyle='-', color='grey', zorder=-1) # Right track limit
        ax.axis('equal')
        return ax

class OvalTrack(Track):
    def __init__(self, track_config):
        super().__init__(track_config)

    def generateTrackRep(self, track_config):
        self.half_width = track_config["track_half_width"]
        straight_len = track_config["straight_length"]
        curve_rad = track_config["curve_radius"]
        ds = track_config["ds"]

        segment_curvature = [0, 1/curve_rad, 0, 1/curve_rad]
        segment_change = np.cumsum([straight_len, curve_rad*np.pi, straight_len, curve_rad*np.pi])

        self.total_len, self.s, self.track_curvature, self.track_xypsi = self.generateTrackFromCurvature(segment_curvature, segment_change, ds) 
        self.name = "OVAL w/Straight {}, Curve Radius {}".format(track_config["straight_length"], track_config["curve_radius"]) 

class LTrack(Track):
    def __init__(self, track_config):
        super().__init__(track_config)

    def generateTrackRep(self, track_config):
        self.half_width = track_config["track_half_width"]
        straight_len = track_config["straight_length"]
        curve_rad = track_config["curve_radius"]
        ds = track_config["ds"]

        segment_curvature = [0, 1/curve_rad, -1/curve_rad, 1/curve_rad, 0, 1/curve_rad]
        segment_change = np.cumsum([straight_len, curve_rad*np.pi, curve_rad*np.pi/2, curve_rad*np.pi, straight_len, curve_rad*np.pi/2])

        self.total_len, self.s, self.track_curvature, self.track_xypsi = self.generateTrackFromCurvature(segment_curvature, segment_change, ds) 
        self.name = "L TRACK w/Straight {}, Curve Radius {}".format(track_config["straight_length"], track_config["curve_radius"]) 

if __name__ == "__main__":
    # Oval Track
    # track_config = {"track_half_width":10, "straight_length":100, "curve_radius":90, "ds":0.05}
    # track = OvalTrack(track_config)

    # L Track
    track_config = {"track_half_width":15, "straight_length":1000, "curve_radius":500, "ds":0.05}
    track = LTrack(track_config)
    
    """
    curvilinear state: [s, ey, epsi, vx, vy, w, delta]
    global state: [x, y, theta, vx, vy, w, delta]
    """
    cl_state = np.array([320, -7, np.pi/4, 15, 15, 0, 0])
    glob_state = track.CLtoGlobal(cl_state)
    track_x,track_y,track_psi = track.getTrackPosition(cl_state[0])
    # print("Curvilinear State", cl_state)
    # print("Global State", glob_state)

    # track.plotTrack()    
    # plt.scatter(track_x, track_y)
    # plt.quiver(glob_state[0], glob_state[1], np.cos(glob_state[2]), np.sin(glob_state[2]))
    # plt.quiver(glob_state[0], glob_state[1], glob_state[3], glob_state[4], color='r')
    # plt.legend(["Centerline", "Left Bound", "Right Bound", "Track pos", "Pose", "Velo"])
    # plt.axis('equal')
    # plt.show()

    # Test get curvature
    # s = np.linspace(0,10000,1000)
    # plt.plot(s,track.getCurvature(s))
    # plt.plot(s,track.getCurvature(s%6712))
    # plt.show()


