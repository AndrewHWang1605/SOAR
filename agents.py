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
    def __init__(self, veh_config, scene_config, x0, controller):
        self.veh_config = veh_config
        self.scene_config = scene_config
        self.x = x0
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
    def __init__(self, veh_config, scene_config, x0, controller):
        super().__init__(veh_config, scene_config, x0, controller)

    """
    Implement dynamics and update state one timestep later
    oppo_states: Nxk 
    """
    def step(self, oppo_states, curvature):
        # accel, delta_dot = self.controller.computeControl(self.x[-1], oppo_states, curvature)
        accel, delta_dot = 0, 0
        x_new = self.dynamics(accel, delta_dot)
        return x_new
    
    # Steps forward dynamics of vehicle one discrete timestep
    def dynamics(self, accel, delta_dot):
        # Expands state variable and precalculates sin/cos
        s, ey, epsi, vx, vy, omega, delta = self.x
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
        Fyf, Fyr = self.lateralForce()
        Fd = 0 #self.dragForce()

        # Calculate x_dot components from dynamics equations
        s_dot = (vx*cos_epsi - vy*sin_epsi) / (1 - ey * kappa)
        ey_dot = vx*sin_epsi + vy*cos_epsi
        epsi_dot = omega - kappa*s_dot
        vx_dot = 1/m * (Fxr - Fd - Fyf*sin_delta + m*vy*omega)
        vy_dot = 1/m * (Fyr + Fyf*cos_delta - m*vx*omega)
        omega_dot = 1/Iz * (lf*Fyf*cos_delta - lr*Fyr)

        print("vx_dot", np.round(vx_dot,4), "Fyf*sin_delta", np.round(Fyf*sin_delta,4), "Fyf*cos_delta", np.round(Fyf*cos_delta,4))

        # Propogate state variable forwards one timestep with Euler step
        x_dot = np.array([s_dot, ey_dot, epsi_dot, vx_dot, vy_dot, omega_dot, delta_dot])
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

        if vx < 1e-3:
            alpha_f, alpha_r = 0,0
        else: 
            if abs((vy + lf*omega) / vx) < 0.1:
                alpha_f = (vx*delta - vy - lf*omega) / vx
            else:
                # alpha_f = delta - np.arctan2((vy + lf*omega), vx)
                alpha_f = delta - np.arctan((vy + lf*omega) / vx)
            
            if abs((vy - lr*omega) / vx) < 0.1:
                alpha_r = (-vy + lr*omega) / vx
            else:
                # alpha_r = -np.arctan2((vy - lr*omega), vx)
                alpha_r = -np.arctan((vy - lr*omega) / vx)
                

        print(alpha_f, alpha_r)
        return alpha_f, alpha_r
    

    # Frontal drag force of vehicle
    def dragForce(self):
        vx = self.x[4]
        rho = 1.225
        Cd = self.veh_config["Cd"]
        SA = self.veh_config["SA"]
        Fd = 0.5 * rho * SA * Cd * vx**2
        return Fd

        
if __name__ == "__main__":
    from track import OvalTrack
    from config import get_vehicle_config, get_scene_config
    import matplotlib.pyplot as plt

    veh_config = get_vehicle_config()
    scene_config = get_scene_config()
    x0 = [0, 0, 0, 2, 0, 0, np.pi/3]
    agent = BicycleVehicle(veh_config, scene_config, x0, 0)
    x_holder = []
    for i in range(25):
        holder = np.round(agent.step(0,0),4)
        # print("CL coords", holder)
        holder = scene_config["track"].CLtoGlobal(holder)
        print(str(i), " Global coords", np.round(holder,2))
        x_holder.append(holder)

    x_holder = np.array(x_holder)
    scene_config["track"].plotTrack()
    plt.plot(x_holder[:, 0], x_holder[:, 1])
    plt.show()
    

