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
        self.x_hist = np.array([x0])
        self.u_hist = None

    # Implement dynamics and update state one timestep later
    def step(self, oppo_states, t):
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
    def step(self, oppo_states, curvature, t):
        accel, delta_dot = self.controller.computeControl(self.x_hist[-1], oppo_states, curvature, t)
        accel, delta_dot = self.saturate_inputs(accel, delta_dot)
        x_new = self.dynamics(accel, delta_dot)
        self.x = x_new
        self.x_hist = np.vstack((self.x_hist, self.x))
        self.u_hist = np.vstack((self.u_hist, np.array([accel, delta_dot]))) if self.u_hist is not None else np.array([accel, delta_dot])
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
        Fd = self.dragForce()

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
        x_new = self.x + x_dot*dt
        x_new[6] = np.clip(x_new[6], -self.veh_config["max_steer"], self.veh_config["max_steer"])
        return x_new

    # Rear wheel drive, all acceleration goes onto rear wheels
    def longitudinalForce(self, accel):
        print(accel)
        m = self.veh_config["m"]
        Fxf, Fxr = 0, m*accel
        return Fxf, Fxr
    
    # Simple linearized lateral forces/tire model with slip angles
    def lateralForce(self):
        c = self.veh_config["c"]
        alpha_f, alpha_r = self.slipAngles()
        Fyf, Fyr = c*alpha_f, c*alpha_r
        return Fyf, Fyr
    
    # Slip angles for tires
    def slipAngles(self):
        vx, vy, omega, delta = self.x[3:]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]
        eps = 1e-6 # Avoid divide by 0

        # alpha_f = delta - np.arctan((vy + lf*omega) / (vx+eps))
        # alpha_r = -np.arctan((vy - lr*omega) / (vx+eps))

        if vx < 1e-3:
            alpha_f, alpha_r = 0,0
        else: 

            if abs((vy + lf*omega) / vx) < 0.1:
                alpha_f = (vx*delta - vy - lf*omega) / vx
            else:
                alpha_f = delta - np.arctan((vy + lf*omega) / vx)
            
            if abs((vy - lr*omega) / vx) < 0.1:
                alpha_r = (-vy + lr*omega) / vx
            else:
                alpha_r = -np.arctan((vy - lr*omega) / vx)


        #TODO: Simplify this?                
        # print("Pre-Slip Angles", vx, vy, omega, delta)
        # print("Slip Angles [deg]", alpha_f/np.pi * 180, alpha_r/np.pi*180)
        return alpha_f, alpha_r
    
    # Frontal drag force of vehicle
    def dragForce(self):
        vx = self.x[4]
        rho = 1.225
        Cd = self.veh_config["Cd"]
        SA = self.veh_config["SA"]
        Fd = 0.5 * rho * SA * Cd * vx**2
        return Fd

    def saturate_inputs(self, accel, delta_dot):
        accel_clipped = np.clip(accel, -self.veh_config["max_accel"], self.veh_config["max_accel"])
        delta_dot_clipped = np.clip(delta_dot, -self.veh_config["max_steer_rate"], self.veh_config["max_steer_rate"])
        return accel_clipped, delta_dot_clipped

if __name__ == "__main__":
    from track import OvalTrack
    from config import get_vehicle_config, get_scene_config, get_controller_config
    import matplotlib.pyplot as plt
    from controllers import SinusoidalController, ConstantVelocityController

    veh_config = get_vehicle_config()
    scene_config = get_scene_config()
    cont_config = get_controller_config()
     
    dt = scene_config["dt"]
    x0 = np.array([0, 0, 0, 12, 0, 0, 0])
    cont = ConstantVelocityController(veh_config, scene_config, cont_config)
    # cont = SinusoidalController(veh_config, scene_config, cont_config)
    agent = BicycleVehicle(veh_config, scene_config, x0, cont)

    t_hist = [0]
    xg_hist = [x0]
    for i in range(50000):
        t = t_hist[-1] + dt
        t_hist.append(t)
        x_cl = agent.step(0, [], t)
        print("CL coords", np.round(x_cl,2), "\n")
        x_g = scene_config["track"].CLtoGlobal(x_cl)
        # print(str(i), " Global coords", np.round(x_g,2))
        xg_hist.append(x_g)
        # if (np.abs(holder[2]) > 5e-2):
        #     break

    def plot_cl_states(t_hist, agent):
        titles = ["s", "ey", "epsi", "vx", "vy", "omega", "delta", "accel", "delta_dot"]
        plt.figure(figsize=(15,8))
        for i in range(7):
            plt.subplot(3,3,i+1)
            plt.plot(t_hist, agent.x_hist[:,i])
            plt.title(titles[i])

        for i in range(7,9):
            plt.subplot(3,3,i+1)
            plt.plot(t_hist[:-1], agent.u_hist[:,i-7])
            plt.title(titles[i])
        # plt.show()

        
    plot_cl_states(t_hist, agent)

    plt.figure()
    scene_config["track"].plotTrack()
    xg_hist = np.array(xg_hist)
    plt.plot(xg_hist[:, 0], xg_hist[:, 1])
    plt.show()
    # plt.scatter(xg_hist[:, 0], xg_hist[:, 1])

    # plt.show()
    

