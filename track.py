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
    def __init__(self):
        self.generateTrackRep()        

    # Convert curvilinear coordinates to global
    def CLtoGlobal(self, state):
        """
        curvilinear state: [s, ey, ephi, vx, vy, w, delta]
        global state: [x, y, theta, vx, vy, w, delta]
        """
        global_state = np.zeros(6)
        return global_state

class OvalTrack(Track):
    def __init__(self):
        super().__init__()

    def generateTrackRep(self):
        track_width = 10
        straight_len = 100
        curve_rad = 10

        # Straight
        segment_curvature = [0, 1/curve_rad, 0, 1/curve_rad]
        segment_change = np.cumsum([straight_len, curve_rad*np.pi, straight_len, curve_rad*np.pi])

        self.total_len = 2*(straight_len + np.pi*curve_rad)

        ds = 0.1
        s = np.arange(0, self.total_len, ds)
        track_curvature = np.zeros(s.shape[0])
        track_xypsi = np.zeros((s.shape[0], 3))
        track_xypsi[0,:] = np.array([curve_rad, 0, 0])
        segment_counter = 0
        for i in range(s.shape[0]):
            if (s[i] > segment_change[segment_counter]):
                segment_counter += 1
            track_curvature[i] = segment_curvature[segment_counter]
            
            if (i < s.shape[0]-1):
                dtheta = track_xypsi[i,2] - np.arcsin(0.5*ds*track_curvature[i])
                track_xypsi[i+1,:] = track_xypsi[i,:] + np.array([ds*np.cos(dtheta), ds*np.sin(dtheta), -2*np.arcsin(0.5*ds*track_curvature[i])])
            else:
                dtheta = track_xypsi[i,2] - np.arcsin(0.5*ds*track_curvature[i])
                loop_back_start = track_xypsi[i,:] + np.array([ds*np.cos(dtheta/2), ds*np.sin(dtheta/2), dtheta])
                print(track_xypsi[0,:])
                print(loop_back_start)
        
        self.track_curvature = track_curvature
        self.track_xypsi = track_xypsi
        self.s = s 

    def getCurvature(self, s):
        nearest_s_ind = np.argmin(np.abs(self.s - s%self.total_len))
        return self.track_curvature[nearest_s_ind]
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ovaltrack = OvalTrack()
    plt.scatter(ovaltrack.track_xypsi[:,0], ovaltrack.track_xypsi[:,1], marker='x')
    plt.axis('equal')
    # print(ovaltrack.track_xypsi[:,2])
    # print(np.max(ovaltrack.track_xypsi[:,2]))

    # plt.plot(ovaltrack.track_xypsi[:,2])
    plt.show()


