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
import casadi as ca
import copy

class Agent:
    def __init__(self, veh_config, scene_config, x0, controller, ID=999):
        self.veh_config = veh_config
        self.scene_config = scene_config
        self.x = x0
        self.controller = controller
        self.ID = ID
        self.current_timestep = 0

        self.init_histories()


    def init_histories(self):
        sim_time = self.scene_config["sim_time"]
        self.dt = self.scene_config["dt"]
        timesteps = int(sim_time / self.dt)

        self.x_hist = np.zeros((timesteps+1, 7))
        self.x_hist[0,:] = self.x
        self.x_global_hist = np.zeros((timesteps+1, 7))
        self.x_global_hist[0,:] = np.array([self.scene_config["track"].CLtoGlobal(self.x)])
        self.u_hist = np.zeros((timesteps, 2))

    # Implement dynamics and update state one timestep later
    def step(self, oppo_states):
        raise NotImplementedError("Inheritance not implemented correctly")

    def getLastState(self):
        return self.x_hist[self.current_timestep]
    
    def getStateHistory(self):
        return self.x_hist[:self.current_timestep+1]
    
    def getGlobalStateHistory(self):
        return self.x_global_hist[:self.current_timestep+1]
    
    def getControlHistory(self):
        return self.u_hist[:self.current_timestep]

    def ID(self):
        return self.ID
    
    @property
    def size(self):
        return self.veh_config["size"]

   
class BicycleVehicle(Agent):
    def __init__(self, veh_config, scene_config, x0, controller, ID=999):
        super().__init__(veh_config, scene_config, x0, controller, ID)

    """
    Implement dynamics and update state one timestep later
    oppo_states: Nxk 
    """
    def step(self, oppo_states):
        accel, delta_dot = self.controller.computeControl(self.x_hist[self.current_timestep], oppo_states, self.current_timestep*self.dt)
        accel, delta_dot = self.saturate_inputs(accel, delta_dot)
        x_new = self.dynamics(self.x, accel, delta_dot)
        self.x = x_new
        self.x_global = self.scene_config["track"].CLtoGlobal(x_new)
        self.x_hist[self.current_timestep+1, :] = copy.deepcopy(self.x)
        self.x_global_hist[self.current_timestep+1, :] = copy.deepcopy(self.x_global)
        self.u_hist[self.current_timestep, :] = np.array([accel, delta_dot])
        self.current_timestep += 1
        return x_new
    
    # Steps forward dynamics of vehicle one discrete timestep
    def dynamics(self, x, accel, delta_dot):
        # Expands state variable and precalculates sin/cos
        s, ey, epsi, vx, vy, omega, delta = x
        sin_epsi, cos_epsi = np.sin(epsi), np.cos(epsi)
        sin_delta, cos_delta = np.sin(delta), np.cos(delta)

        # Expand scene and vehicle config variables
        dt = self.scene_config["dt"]
        kappa = self.scene_config["track"].getCurvature(s)
        m = self.veh_config["m"]
        Iz = self.veh_config["Iz"]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]

        # Calculate various forces 
        Fxf, Fxr = self.longitudinalForce(accel)
        Fyf, Fyr = self.lateralForce(x)
        Fxf, Fxr, Fyf, Fyr = self.saturate_forces(Fxf, Fxr, Fyf, Fyr)
        Fd = self.dragForce(x)

        # Calculate x_dot components from dynamics equations
        s_dot = (vx*cos_epsi - vy*sin_epsi) / (1 - ey * kappa)
        ey_dot = vx*sin_epsi + vy*cos_epsi
        epsi_dot = omega - kappa*s_dot
        vx_dot = 1/m * (Fxr - Fd - Fyf*sin_delta + m*vy*omega)
        vy_dot = 1/m * (Fyr + Fyf*cos_delta - m*vx*omega)
        omega_dot = 1/Iz * (lf*Fyf*cos_delta - lr*Fyr)

        # Propogate state variable forwards one timestep with Euler step
        x_dot = np.array([s_dot, ey_dot, epsi_dot, vx_dot, vy_dot, omega_dot, delta_dot])
        # print("xdot", np.round(x_dot, 4))
        x_new = x + x_dot*dt
        x_new[6] = np.clip(x_new[6], -self.veh_config["max_steer"], self.veh_config["max_steer"])
        return x_new

    # Steps forward dynamics of vehicle one discrete timestep for CasADi symbolic vars
    def casadi_dynamics(self, x, accel, delta_dot, kappa):
        # Expands state variable and precalculates sin/cos
        s, ey, epsi, vx, vy, omega, delta = [x[i] for i in range(self.x.shape[0])]
        sin_epsi, cos_epsi = ca.sin(epsi), ca.cos(epsi)
        sin_delta, cos_delta = ca.sin(delta), ca.cos(delta)

        # Expand scene and vehicle config variables
        dt = self.scene_config["dt"]
        m = self.veh_config["m"]
        Iz = self.veh_config["Iz"]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]

        # Calculate various forces 
        Fxf, Fxr = self.longitudinalForce(accel)
        Fyf, Fyr = self.lateralForce(x)
        # Fxf, Fxr, Fyf, Fyr = self.saturate_forces(Fxf, Fxr, Fyf, Fyr)
        Fd = self.dragForce(x)

        # Calculate x_dot components from dynamics equations
        s_dot = (vx*cos_epsi - vy*sin_epsi) / (1 - ey * kappa)
        ey_dot = vx*sin_epsi + vy*cos_epsi
        epsi_dot = omega - kappa*s_dot
        vx_dot = 1/m * (Fxr - Fd - Fyf*sin_delta + m*vy*omega)
        vy_dot = 1/m * (Fyr + Fyf*cos_delta - m*vx*omega)
        omega_dot = 1/Iz * (lf*Fyf*cos_delta - lr*Fyr)

        # Propogate state variable forwards one timestep with Euler step
        x_dot = ca.SX([s_dot, ey_dot, epsi_dot, vx_dot, vy_dot, omega_dot, delta_dot])
        # print("xdot", np.round(x_dot, 4))
        x_new = x + x_dot*dt
        # x_new[6] = np.clip(x_new[6], -self.veh_config["max_steer"], self.veh_config["max_steer"])
        return x_new

    # Rear wheel drive, all acceleration goes onto rear wheels
    def longitudinalForce(self, accel):
        m = self.veh_config["m"]
        Fxf, Fxr = 0, m*accel
        return Fxf, Fxr
    
    # Simple linearized lateral forces/tire model with slip angles
    def lateralForce(self, x):
        c = self.veh_config["c"]
        alpha_f, alpha_r = self.slipAngles(x)
        Fyf, Fyr = c*alpha_f, c*alpha_r
        return Fyf, Fyr
    
    # Slip angles for tires
    def slipAngles(self, x):
        vx, vy, omega, delta = x[3:]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]
        eps = 1e-6 # Avoid divide by 0

        alpha_f = delta - np.arctan((vy + lf*omega) / (vx+eps))
        alpha_r = -np.arctan((vy - lr*omega) / (vx+eps))
                 
        # print("Pre-Slip Angles", vx, vy, omega, delta)
        # print("Slip Angles [deg]", alpha_f/np.pi * 180, alpha_r/np.pi*180)
        return alpha_f, alpha_r
    
    # Frontal drag force of vehicle
    def dragForce(self, x):
        vx = x[4]
        rho = 1.225
        Cd = self.veh_config["Cd"]
        SA = self.veh_config["SA"]
        Fd = 0.5 * rho * SA * Cd * vx**2
        return Fd

    def saturate_inputs(self, accel, delta_dot):
        accel_clipped = np.clip(accel, -self.veh_config["max_accel"], self.veh_config["max_accel"])
        delta_dot_clipped = np.clip(delta_dot, -self.veh_config["max_steer_rate"], self.veh_config["max_steer_rate"])
        return accel_clipped, delta_dot_clipped

    def saturate_forces(self, Fxf, Fxr, Fyf, Fyr):
        F = np.array([Fxf, Fxr, Fyf, Fyr])
        Fmax = self.veh_config["downforce_coeff"] * self.veh_config["m"] * 9.81
        if np.linalg.norm(F) < Fmax:
            return Fxf, Fxr, Fyf, Fyr

        else:
            print("Exceeded max force: {}g".format(np.linalg.norm(F)/Fmax))
            return Fxf, Fxr, Fyf, Fyr

            # normF = F / np.linalg.norm(F) * Fmax
            # return normF[0], normF[1], normF[2], normF[3]
    
