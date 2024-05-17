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
Implement various agents 
"""
import numpy as np


class Agent:
    def __init__(self, veh_config, scen_config, x0, controller):
        self.veh_config = veh_config
        self.scen_config = scen_config
        self.x = [x0]
        self.controller = controller

    # Implement dynamics and update state one timestep later
    def step(self, oppo_states):
        raise NotImplementedError("Inheritance not implemented correctly")

    def getLastState(self):
        return self.x[-1]

    @property
    def ID(self):
        return self.config["ID"] 
    
    @property
    def size(self):
        return self.congif["size"]

   
class BicycleVehicle(Agent):
    def __init__(self, veh_config, scen_config, x0, controller):
        super().__init__(veh_config, scen_config, x0, controller)

    """
    Implement dynamics and update state one timestep later
    oppo_states: Nxk 
    """
    def step(self, oppo_states, curvature):
        accel, delta_dot = self.controller.computeControl(self.x[-1], oppo_states, curvature)
        x_new = self.dynamics(accel, delta_dot)
        return x_new
        
    def dynamics(self, accel, delta_dot):
        s, ey, epsi, vx, vy, omega, delta = self.x
        sin_epsi, cos_epsi = np.sin(epsi), np.cos(epsi)
        sin_delta, cos_delta = np.sin(delta), np.cos(delta)

        dt = self.scen_config["dt"]
        kappa = self.scen_config["track"].getCurvature(s)
        m = self.veh_config["m"]
        Iz = self.veh_config["Iz"]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]

        Fxf, Fxr = self.longitudinalForce(accel)
        Fyf, Fyr = self.lateralForce()
        Fd = self.dragForce(vx)

        s_dot = (vx*cos_epsi - vy*sin_epsi) / (1 - ey * kappa)
        ey_dot = vx*sin_epsi + vy*cos_epsi
        ephi_dot = omega - kappa*s_dot
        vx_dot = 1/m * (Fxr - Fd - Fyf*sin_delta + Fxf*cos_delta + m*vy*omega)
        vy_dot = 1/m * (Fyr + Fyf*cos_delta + Fxf*sin_delta - m*vx*omega)
        omega_dot = 1/Iz * (lf * (Fyf*cos_delta + Fxf*sin_delta) - lr*Fyr)

        x_dot = [s_dot, ey_dot, ephi_dot, vx_dot, vy_dot, omega_dot, delta_dot]
        self.x = self.x + x_dot*dt
        return self.x

    
    # Rear wheel drive, all acceleration goes onto rear wheels
    def longitudinalForce(self, accel):
        m = self.veh_config["m"]
        Fxf, Fxr = 0, m*accel
        return Fxf, Fxr
    
    # Simple linearized lateral forces/tire model with slip angles
    def lateralForce(self):
        c = self.veh_config["c"]
        alpha_f, alpha_r = self.slipAngles()
        Fyf, Fyr = -c*alpha_f, -c*alpha_r
        return Fyf, Fyr
        
    # Slip angles for tires
    def slipAngles(self):
        vx, vy, omega, delta = self.x[3:]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]

        alpha_f = np.atan2((vy + lf*omega) / vx) - delta
        alpha_r = np.atan2((vy - lr*omega) / vx)
        return alpha_f, alpha_r

    # Frontal drag force of vehicle
    def dragForce(self):
        vx = self.x[4]
        rho = 1.225
        Cd = self.veh_config["Cd"]
        SA = self.veh_config["SA"]

        Fd = 0.5 * rho * SA * Cd * vx**2
        return Fd

        
        
        

