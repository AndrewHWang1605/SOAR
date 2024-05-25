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
Implement configs
"""

from track import OvalTrack, LTrack
import numpy as np

OVAL_TRACK = 0
L_TRACK = 1


def get_vehicle_config():
    veh_config = {}

    veh_config["m"] = 800 #2            # kg mass
    veh_config["Cd"] = 0.56             # drag coeff
    veh_config["SA"] = 2                # m^2 frontal SA
    veh_config["Iz"] = 1800 #0.03        # kg/m^2 
    veh_config["lf"] = 2.0 #0.125       # m length forward from CoM
    veh_config["lr"] = 2.0 #0.125       # m length backward from CoM
    veh_config["size"] = 4.1
    veh_config["downforce_coeff"] = 5
    # veh_config["R"] = 0.5               # m radius of tire

    veh_config["max_accel"] = 10 # m/s^2 Max acceleration (assumed symmetric accel/brake)
    veh_config["max_steer_rate"] = 3  # rad/s Max steering rate 
    veh_config["max_steer"] = 0.5 # rad Steering Lock

    # TODO: Confirm/Change
    veh_config["c"] = 210000  #46         # N/rad wheel stiffness 
    # https://www.racecar-engineering.com/tech-explained/tyre-dynamics/

    return veh_config




def get_vehicle_opt_constraints(veh_config, scene_config):
    veh_constraints = {}

    veh_constraints["lb_s"] = 0.0 # minimum longitudinal position
    veh_constraints["lb_ey"] = -scene_config["track_config"]["track_half_width"] # minimum lateral error
    veh_constraints["lb_epsi"] = -10*np.pi/180 # rad minimum heading error
    veh_constraints["lb_vx"] = 0 # m/s minimum longitudinal velocity
    veh_constraints["lb_vy"] = -2 # m/s minimum lateral velocity
    veh_constraints["lb_omega"] = -1 # rad/s minimum angular velocity
    veh_constraints["lb_delta"] = -veh_config["max_steer"] # rad minimum angular velocity

    veh_constraints["ub_s"] = 1.2*scene_config["track"].total_len # maximum longitudinal position
    veh_constraints["ub_ey"] = scene_config["track_config"]["track_half_width"] # maximum lateral error
    veh_constraints["ub_epsi"] = 10*np.pi/180 # rad maximum heading error
    veh_constraints["ub_vx"] = 10 # m/s maximum longitudinal velocity
    veh_constraints["ub_vy"] = 2 # m/s maximum lateral velocity
    veh_constraints["ub_omega"] = 1 # rad/s maximum angular velocity
    veh_constraints["ub_delta"] = veh_config["max_steer"] # rad/s maximum angular velocity

    return veh_constraints




def get_scene_config(track_type=OVAL_TRACK):
    scene_config = {}

    if track_type == OVAL_TRACK:
        # track_config = {"track_half_width":10, "straight_length":100, "curve_radius":90, "ds":0.05}
        track_config = {"track_half_width":10, "straight_length":1000, "curve_radius":250, "ds":0.1}
        track = OvalTrack(track_config)
    elif track_type == L_TRACK:
        track_config = {"track_half_width":15, "straight_length":100, "curve_radius":50, "ds":0.05}
        track = LTrack(track_config)

    scene_config["track"] = track
    scene_config["track_config"] = track_config
    scene_config["dt"] = 0.005
    scene_config["sim_time"] = 100

    return scene_config




def get_controller_config():
    controller_config = {}

    # SINUSOIDAL
    controller_config["omega"] = 1.5
    
    # PID CONSTANT VELOCITY
    # controller_config["k_v"] = [0.5, 1e-3, 2e-1]
    # controller_config["k_theta"] = [2e-2, 2e-4, 8e0]
    # controller_config["k_delta"] = [1e1, 4e0, 7e-1]
    controller_config["k_v"] = [6e1, 4e0, 2e1]
    controller_config["k_theta"] = [3e0, 3e-2, 1e1]
    controller_config["k_delta"] = [3e1, 3e-1, 1e2]

    return controller_config
