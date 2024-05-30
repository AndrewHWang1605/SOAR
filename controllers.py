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

STATE_DIM = 7
INPUT_DIM = 2

class Controller:
    def __init__(self, veh_config, scene_config, control_config):
        self.veh_config = veh_config
        self.scene_config = scene_config
        self.control_config = control_config

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        raise NotImplementedError("Inheritance not implemented correctly")

"""Sinusoidal steering with const acceleration (for debugging)"""
class SinusoidalController(Controller):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        w = self.control_config["omega"]
        return 2, w*np.pi/180*np.cos(w*t)

"""Constant velocity controller for trajectory following of track centerline"""
class ConstantVelocityController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, v_ref=12):
        super().__init__(veh_config, scene_config, control_config)
        self.v_ref = v_ref
        self.prev_v_error = 0
        self.total_v_error = 0
        self.delta_prev = 0
        self.prev_theta_error = 0
        self.total_theta_error = 0

        self.delta_error = 0
        self.prev_delta_error = 0
        self.total_delta_error = 0

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """

        """Unpack global config variables and current state variables"""
        accel, steering_rate = 0, 0
        s, ey, epsi = state[:3]
        track = self.scene_config["track"]
        dt = self.scene_config["dt"]
        lf, lr = self.veh_config["lf"], self.veh_config["lr"]
        k_v, k_theta, k_delta = self.control_config["k_v"], self.control_config["k_theta"], self.control_config["k_delta"]

        """Caclulate global position and expected track position/curvature"""
        global_state = track.CLtoGlobal(state)
        x, y, theta, vx, vy, w, delta = global_state
        track_position = track.getTrackPosition(s)
        x_track, y_track, theta_track = track_position
        kappa = track.getCurvature(s)

        """PID control for velocity with acceleration input with anti-windup"""
        v = np.linalg.norm([vx, vy])
        v_error = (self.v_ref - v) 
        accel = (k_v[0] * v_error) + (k_v[1] * self.total_v_error) + (k_v[2] * (v_error - self.prev_v_error))
        if (np.abs(accel) < self.veh_config["max_accel"]):
            self.total_v_error += v_error*dt
        self.prev_v_error = v_error
        accel = np.clip(accel, -self.veh_config["max_accel"], self.veh_config["max_accel"])

        """Pre-calculating heading theta error with overlap check"""
        theta_error = theta_track - theta
        if (theta_track > 13/14*np.pi and theta < 1/14*np.pi):
            theta_error -= 2*np.pi
        elif (theta_track < 1/14*np.pi and theta > 13/14*np.pi):
            theta_error += 2*np.pi

        """Pre-calculating steering delta error with overlap check"""
        beta = np.arcsin(lr * kappa)
        delta_des = np.arctan((lf+lr)/lr * np.tan(beta))
        delta_error = delta_des - delta
        if (delta_des > 13/14*np.pi and delta < 1/14*np.pi):
            delta_error -= 2*np.pi
        elif (delta_des < 1/14*np.pi and delta > 13/14*np.pi):
            delta_error += 2*np.pi
        
        """
        PID control for steering with steering rate input with anti-windup
        Switches between theta error and delta error based on ey
        """
        if (ey > 0 and theta_error < 0) or (ey < 0 and theta_error > 0):
            theta_dot = (k_theta[0] * theta_error) + (k_theta[1] * self.total_theta_error) + (k_theta[2] * (theta_error - self.prev_theta_error))
            if (np.abs(theta_dot) < self.veh_config["max_steer_rate"]):
                self.total_theta_error += theta_error*dt
            self.prev_theta_error = theta_error
            # print("theta_control", theta_dot)
            steering_rate = theta_dot
        else:
            delta_dot = (k_delta[0] * (delta_error)) + (k_delta[1] * self.total_delta_error) + (k_delta[2] * (delta_error - self.prev_delta_error))
            if (np.abs(delta_dot) < self.veh_config["max_steer_rate"]):
                self.total_delta_error += delta_error*dt
            self.prev_delta_error = delta_error
            # print("delta_control", delta_dot)
            steering_rate = delta_dot
        steering_rate = np.clip(steering_rate, -self.veh_config["max_steer_rate"], self.veh_config["max_steer_rate"])

        return accel, steering_rate
    
"""Try to exactly track nominal trajectory (for debugging)"""
class NominalOptimalController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, raceline_file):
        super().__init__(veh_config, scene_config, control_config)
        unpack_file = np.load(raceline_file)
        self.s_hist = unpack_file["s"]
        self.accel_hist = unpack_file["u_a"]
        self.ddelta_hist = unpack_file["u_s"]

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        s, ey, epsi, vx_cl, vy_cl, w, delta = state
        total_len = self.scene_config["track"].total_len
        s = np.mod(np.mod(s, total_len) + total_len, total_len)
        nearest_s_ind = np.where(s >= self.s_hist)[0][-1]
        print(nearest_s_ind)
        return self.accel_hist[nearest_s_ind], self.ddelta_hist[nearest_s_ind]

"""Try to exactly track nominal trajectory (for debugging)"""
class MPCController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, raceline_file):
        super().__init__(veh_config, scene_config, control_config)
        self.race_line = np.load(raceline_file)
        self.race_line_mat = self.construct_race_line_mat(self.race_line)
        
    def construct_race_line_mat(self, raceline):
        """
        Unpack raceline dictionary object into matrix 
        Returns 10xN (0:time, 1-7: 7 curvilinear states, 8-9: 2 inputs)
        """
        race_line_mat = np.zeros((STATE_DIM+INPUT_DIM+1, self.race_line["t"].shape[0]))
        race_line_mat[0, :] = raceline["t"]
        race_line_mat[1, :] = raceline["s"]
        race_line_mat[2, :] = raceline["e_y"]
        race_line_mat[3, :] = raceline["e_psi"]
        race_line_mat[4, :] = raceline["v_long"]
        race_line_mat[5, :] = raceline["v_tran"]
        race_line_mat[6, :] = raceline["psidot"]
        race_line_mat[7, :] = raceline["delta"]
        race_line_mat[8, :] = raceline["u_a"]
        race_line_mat[9, :] = raceline["u_s"]
        
        return race_line_mat


    def init_solver(self):
        pass

    def get_reference_trajectory(self, s, delta_t):
        """
        s: Current longitudinal position
        t: Monotonically increasing vector of time intervals (starting from 0) that we 
           want to evaluate reference trajectory for (eg [0, 0.25, 0.5])
        """
        ref_traj = np.zeros((STATE_DIM+INPUT_DIM, delta_t.shape[0]))

        # Find closest point on reference trajectory and corresponding time
        total_len = self.scene_config["track"].total_len
        s = np.mod(np.mod(s, total_len) + total_len, total_len)
        s_hist = self.race_line_mat[1,:]
        nearest_s_ind = np.where(s >= s_hist)[0][-1]
        closest_t = self.race_line_mat[0, nearest_s_ind]

        t_hist = closest_t + delta_t

        for i in range(ref_traj.shape[0]):
            ref_traj[i,:] = np.interp(t_hist, self.race_line_mat[0,:], self.race_line_mat[i+1,:])

        return ref_traj

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        # s, ey, epsi, vx_cl, vy_cl, w, delta = state
        # total_len = self.scene_config["track"].total_len
        # s = np.mod(np.mod(s, total_len) + total_len, total_len)
        # nearest_s_ind = np.where(s >= self.s_hist)[0][-1]
        # print(nearest_s_ind)
        # return self.accel_hist[nearest_s_ind], self.ddelta_hist[nearest_s_ind]
        return 0, 0

from config import *
veh_config = get_vehicle_config()
scene_config = get_scene_config(track_type=OVAL_TRACK)
cont_config = get_controller_config()
controller = MPCController(veh_config, scene_config, cont_config, "race_lines/oval_raceline.npz")
controller.get_reference_trajectory(3.5, np.array([0,0.25,0.5]))