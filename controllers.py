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
Implement various controllers
"""
import numpy as np

class Controller:
    def __init__(self, veh_config, scene_config, control_config):
        self.veh_config = veh_config
        self.scene_config = scene_config
        self.control_config = control_config

    def computeControl(self, state, oppo_states, curvature, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        raise NotImplementedError("Inheritance not implemented correctly")



"""Sinusoidal steering with const acceleration (for debugging)"""
class SinusoidalController(Controller):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)

    def computeControl(self, state, oppo_states, curvature, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        w = self.control_config["omega"]
        return 2, w*np.pi/180*np.cos(w*t)


class ConstantVelocityController(Controller):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)
        self.v_ref = 12
        self.prev_v_error = 0
        self.total_v_error = 0
        self.delta_prev = 0
        self.prev_theta_error = 0
        self.total_theta_error = 0

        self.delta_error = 0
        self.prev_delta_error = 0
        self.total_delta_error = 0

    def computeControl(self, state, oppo_states, curvature, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        accel, delta_dot = 0, 0
        s, ey, epsi = state[:3]
        track = self.scene_config["track"]
        dt = self.scene_config["dt"]
        lf, lr = self.veh_config["lf"], self.veh_config["lr"]
        k_v, k_theta, k_delta = self.control_config["k_v"], self.control_config["k_theta"], self.control_config["k_delta"]

        global_state = track.CLtoGlobal(state)
        x, y, theta, vx, vy, w, delta = global_state
        track_position = track.getTrackPosition(s)
        x_track, y_track, theta_track = track_position
        kappa = track.getCurvature(s)
        radius = 1/kappa

        v = np.linalg.norm([vx, vy])
        v_error = (self.v_ref - v) / dt
        accel = (k_v[0] * v_error) + (k_v[1] * self.total_v_error) + (k_v[2] * (v_error - self.prev_v_error))
        self.prev_v_error = v_error
        self.total_v_error += v_error

        if (theta_track > 13/14*np.pi and theta < 1/14*np.pi):
            theta_error = theta_track - theta - 2*np.pi
        elif (theta_track < 1/14*np.pi and theta > 13/14*np.pi):
            theta_error = theta_track - theta + 2*np.pi
        else:
            theta_error = theta_track - theta

        if (ey > 0 and theta_error < 0) or (ey < 0 and theta_error > 0):
            print("theta_control")
            self.total_theta_error += theta_error
            theta_dot = (k_theta[0] * theta_error) + (k_theta[1] * self.total_theta_error) + (k_theta[2] * (theta_error - self.prev_theta_error))
            self.prev_theta_error = theta_error
            return accel, theta_dot
        else:
            print("delta_control")
            beta = np.arcsin(lr/radius)
            delta_des = np.arctan((lf+lr)/lr * np.tan(beta))
            delta_error = delta_des - delta
            self.total_delta_error += delta_error
            delta_dot = (k_delta[0] * (delta_error)) + (k_delta[1] * self.total_delta_error) + (k_delta[2] * (delta_error - self.prev_delta_error))
            self.prev_delta_error = delta_error
            return accel, delta_dot
    
        

if __name__ == "__main__":
    from config import get_vehicle_config, get_scene_config, get_controller_config
    veh_config = get_vehicle_config()
    scene_config = get_scene_config()
    control_config = get_controller_config()
    cont = SinusoidalController(veh_config, scene_config, control_config)
    # print(cont.computeControl([1,2,3],[[1,2,3]], 0.2))
