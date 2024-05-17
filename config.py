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

def get_vehicle_config():
    
    veh_config = {}

    veh_config["m"] = 800       # kg mass
    veh_config["Cd"] = 0.56     # drag coeff
    veh_config["SA"] = 2        # m^2 frontal SA
    veh_config["Iz"] = 600      # kg/m^2 
    veh_config["lf"] = 1.5      # m length forward from CoM
    veh_config["lr"] = 1.0      # m length backward from CoM
    veh_config["R"] = 0.5       # m radius of tire

    # TODO: Confirm/Change
    veh_config["c"] = 46        # N/rad wheel stiffness 

    return veh_config


def get_scene_config():

    scene_config = {}

    return scene_config


def get_controller_config():

    controller_config = {}

    return controller_config
