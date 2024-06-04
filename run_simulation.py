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
Implement simulator
"""

import numpy as np
import matplotlib.pyplot as plt

from agents import BicycleVehicle
from track import *
from config import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from controllers import *


class Simulator:
    def __init__(self, scene_config):
        self.scene_config = scene_config
        self.agents = []
        self.sim_time = self.scene_config["sim_time"]
        self.dt = self.scene_config["dt"]
        self.t_hist = np.arange(0, self.sim_time+self.dt, self.dt)
        self.sim_success = True
        self.collision_agents = []

    @property
    def track(self):
        return self.scene_config["track"]
    
    def addAgent(self, agent):
        self.agents.append(agent)

    def runSim(self, end_plot=False, animate=False, save=False, follow_agent_IDs=[None]):
        print("\nStarting simulation at dt=" + str(self.dt) + " for " + str(self.sim_time) + " seconds")

        retVal = True
        sim_steps = int(self.sim_time / self.dt)
        for i in range(sim_steps):
            if i%500 == 0:
                print("Running simulation: ", i*self.dt, " sec passed")

            agent_states = {}
            for agent in self.agents:
                agent_states[agent.ID] = agent.getLastState()

            collision, collision_agents = self.checkCollisions()
            if collision:
                print("Collision detected for agents:", *collision_agents)
                self.sim_success = False
                self.collision_agents = np.array(collision_agents)
                retVal = False
                break

            for agent in self.agents:
                if agent.controller.ctrl_period == None: # Run at every timestep
                    recompute_ctrl = True 
                elif np.isclose((i*self.dt / agent.controller.ctrl_period), np.round(i*self.dt / agent.controller.ctrl_period), atol=1e-3): # Run at proper frequency
                    recompute_ctrl = True 
                else:
                    recompute_ctrl = False
                # print(agent.ID)
                temp = agent_states.pop(agent.ID) # Remove own state to only contain opponents
                agent.step(agent_states, recompute_control=recompute_ctrl)
                agent_states[agent.ID] = temp

        self.t_hist = self.t_hist[:i+2] # Trim off extra timesteps

        if self.sim_success:
            print("Finished simulation: ", sim_steps, " timesteps passed\n")
        if end_plot:
            self.plotCLStates()
            self.plotAgentTrack()
            plt.show()
        if animate:
            for follow_ID in follow_agent_IDs:
                anim = self.animateRace(follow_agent_ID=follow_ID)
                if save:
                    writergif = animation.PillowWriter(fps=30)
                    anim.save('filename.gif',writer=writergif)
                    # anim.save("./videos/race_video_{}.mp4".format("agent"+str(follow_ID) if follow_ID is not None else "global"))
                else:
                    plt.show()
        return retVal
    

    """Check for collisions between all agents"""
    def checkCollisions(self):
        collision = False
        collision_agents = set()

        # Iterate through all agents in double loop
        for agent in self.agents:
            agent_state_CL = agent.getLastState()
            for oppo_agent in self.agents:
                if agent.ID == oppo_agent.ID:
                    break
                oppo_agent_state_CL = oppo_agent.getLastState()

                # Convert agent states to global frame from curvilinear
                track = self.scene_config["track"]
                agent_state_global = track.CLtoGlobal(agent_state_CL)
                oppo_agent_state_global = track.CLtoGlobal(oppo_agent_state_CL)

                # Check agent collision along long/lat axes, add agent IDs to set if collision
                if (agent_state_CL[0] < oppo_agent_state_CL[0]): # Agent behind opponent
                    long_bound = oppo_agent.lr + agent.lf
                else:
                    long_bound = oppo_agent.lf + agent.lr
                lat_bound = oppo_agent.halfwidth + agent.halfwidth
                collision_state = np.abs(agent_state_CL[:2] - oppo_agent_state_CL[:2]) <= np.array([long_bound, lat_bound])
                if np.all(collision_state):
                    print(agent_state_CL-oppo_agent_state_CL)
                    collision = True
                    collision_agents.update([agent.ID, oppo_agent.ID])

        return collision, list(collision_agents)

    def addMeasurementNoise():
        # TODO: Implement!
        pass
    

    def exportSimData(self):
        # Ensure entire array is printed in string to csv
        np.set_printoptions(threshold=np.inf)

        sim_data = {}

        sim_data["track_config"] = self.scene_config["track_config"]
        sim_data["sim_success"] = self.sim_success
        sim_data["collision_agents"] = np.array2string(np.array(self.collision_agents), separator=',', suppress_small=True)
        # sim_data["t"] = np.array2string(self.t_hist, separator=',', suppress_small=True)
        sim_data["agent_count"] = len(self.agents)
        for agent in self.agents:
            sim_data["x" + str(agent.ID)] = np.array2string(agent.getStateHistory(), separator=',', suppress_small=True)
            # sim_data["u" + str(agent.ID)] = np.array2string(agent.getControlHistory(), separator=',', suppress_small=True)

        return sim_data


    def plotCLStates(self):
        titles = ["s", "ey", "epsi", "vx", "vy", "omega", "delta", "accel", "delta_dot"]
        plt.figure(0, figsize=(15,8))
        for agent in self.agents:
            x_hist = agent.getStateHistory()
            u_hist = agent.getControlHistory()
            for i in range(7):
                plt.subplot(3,3,i+1)
                plt.plot(self.t_hist[:x_hist.shape[0]], x_hist[:,i])
                plt.title(titles[i])
            for i in range(7,9):
                plt.subplot(3,3,i+1)
                plt.plot(self.t_hist[:u_hist.shape[0]], u_hist[:,i-7])
                plt.title(titles[i])
        plt.legend([str(agent.ID) for agent in self.agents])


    def plotAgentTrack(self):
        self.scene_config["track"].plotTrack()
        for agent in self.agents:
            x_global_hist = agent.getGlobalStateHistory()
            plt.scatter(x_global_hist[0, 0], x_global_hist[0, 1], marker='D')
            plt.plot(x_global_hist[:, 0], x_global_hist[:, 1], label=str(agent.ID))


    def animateRace(self, follow_agent_ID=None):
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]
        yaw = [0.0, 0.5, 1.3, 0.5]
        fig = plt.figure(figsize=(9,9))
        # plt.grid()
        ax = fig.add_subplot(111)
        self.scene_config["track"].plotTrack(ax=ax)

        def center2xy(xc, yc, psi, lf, lr, hw):
            """ Params are center x, center y, orientation, lf, lr, half width of car """
            ll_x = xc - lr*np.cos(psi) + hw*np.sin(psi)
            ll_y = yc - lr*np.sin(psi) - hw*np.cos(psi)
            return ll_x, ll_y

        def init():
            car_patch_list = []
            for agent in self.agents:
                lf = agent.lf
                lr = agent.lr
                hw = agent.halfwidth

                x, y, theta, vx, vy, w, delta = agent.x_global_hist[0,:]
                ll_x, ll_y = center2xy(x, y, theta, lf, lr, hw)
                patch = patches.Rectangle((ll_x, ll_y), lf+lr, 2*hw, fc=agent.color, angle=theta)
                ax.add_patch(patch)
                agent.assignPatch(patch)
                car_patch_list.append(patch)

                if agent.controller.controller_type == "safe_mpc":
                    patchDict = {}
                    print(agent.controller.agentID2ind)
                    for agentID in agent.controller.agentID2ind:
                        traj, = ax.plot([0],[0], agent.color)
                        patchDict[agentID] = traj
                        car_patch_list.append(traj)
                    agent.controller.assignGPPredPatch(patchDict)
            return car_patch_list

        def animate(i):
            changed_patches_list = []
            for agent in self.agents:
                lf = agent.lf
                lr = agent.lr
                hw = agent.halfwidth
                patch = agent.patch

                controller = agent.controller
                if np.isclose((i*self.scene_config["anim_downsample_factor"]*self.dt / agent.controller.ctrl_period), np.round(i*self.scene_config["anim_downsample_factor"]*self.dt / agent.controller.ctrl_period), atol=1e-3): # Run at proper frequency
                    if agent.controller.controller_type == "safe_mpc":
                        gp_pred_hist = agent.controller.gp_pred_hist[int(np.round(i*self.scene_config["anim_downsample_factor"]*self.dt / agent.controller.ctrl_period))]
                        for agentID in controller.agentID2ind:
                            s_ey = gp_pred_hist[controller.agentID2ind[agentID]]
                            xy = self.scene_config["track"].CLtoGlobalPos(s_ey)
                            traj = agent.controller.patchDict[agentID]
                            traj.set_data(xy)

                x, y, theta, vx, vy, w, delta = agent.x_global_hist[i*self.scene_config["anim_downsample_factor"],:]

                ll_x, ll_y = center2xy(x, y, theta, lf, lr, hw)
                if follow_agent_ID is not None and agent.ID == follow_agent_ID:
                    window = self.scene_config["anim_window"]
                    ax.axis([ll_x-window, ll_x+window,ll_y-window, ll_y+window])

                patch.set_xy([ll_x, ll_y])
                patch.set_angle(np.rad2deg(theta))
                changed_patches_list.append(patch)

            return changed_patches_list

        anim = animation.FuncAnimation(fig, animate,
                                    init_func=init,
                                    frames=self.t_hist.shape[0]//self.scene_config["anim_downsample_factor"],
                                    interval=self.scene_config["anim_downsample_factor"] * self.scene_config["dt"] * 1000,
                                    repeat=False,
                                    blit=False)
        # plt.show()
        return anim

if __name__ == "__main__":
    """Initialize configurations"""
    veh_config = get_vehicle_config()
    scene_config = get_scene_config(track_type=L_TRACK)
    cont_config = get_controller_config(veh_config, scene_config)
     
    sim = Simulator(scene_config)
    
    # Stationary obstacle
    x0_1 = np.array([20, 0, 0, 0, 0, 0, 0])
    controller1 = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=0)
    agent1 = BicycleVehicle(veh_config, scene_config, x0_1, controller1, 1, color='b')
    # sim.addAgent(agent1)

    # Max speed PID controller
    x0_2 = np.array([0, 0, 0, 0, 0, 0, 0]) # Qualifying lap
    controller2 = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=75)
    agent2 = BicycleVehicle(veh_config, scene_config, x0_2, controller2, 2, color='r')
    sim.addAgent(agent2)

    x0_3 = np.array([-30, 0, 0, 30, 0, 0, 0])
    # controller3 = ConstantVelocityController(veh_config, scene_config, cont_config)
    # controller3 = NominalOptimalController(veh_config, scene_config, cont_config, "race_lines/oval_raceline.npz")
    controller3 = MPCController(veh_config, scene_config, cont_config)
    # controller3 = AdversarialMPCController(veh_config, scene_config, cont_config)
    agent3 = BicycleVehicle(veh_config, scene_config, x0_3, controller3, 3, color='g')
    # sim.addAgent(agent3)

    # x0_4 =  np.array([920, 0, 0, 10, 0, 0, 0]) # Nice overtake
    # x0_4 =  np.array([650, 0, 0, 40, 0, 0, 0])  # Faster overtake
    # x0_4 =  np.array([0, 0, 0, 10, 0, 0, 0]) # Straight overtake
    x0_4 =  np.array([-50, 5, 0, 0, 0, 0, 0]) # Experimenting
    controller4 = SafeMPCController(veh_config, scene_config, cont_config)
    # controller4 = MPCController(veh_config, scene_config, cont_config)
    agent4 = BicycleVehicle(veh_config, scene_config, x0_4, controller4, 4, color='g', add_noise=False)
    # sim.addAgent(agent4)

    # x0_5 = np.array([960, 0, 0, 10, 0, 0, 0]) # Nice overtake
    # x0_5 = np.array([725, 0, 0, 40, 0, 0, 0])  # Faster overtake
    # x0_5 =  np.array([70, 10, 0, 10, 0, 0, 0]) # Straight overtake
    x0_5 =  np.array([125, 0, 0, 80, 0, 0, 0]) # Experimenting
    controller5 = AdversarialMPCController(veh_config, scene_config, cont_config)
    agent5 = BicycleVehicle(veh_config, scene_config, x0_5, controller5, 5, color='r')
    # sim.addAgent(agent5)

    # x0_6 = np.array([300, -12, 0, 5, 0, 0, 0])
    # # x0_5 = np.array([1000, -5, 0, 5, 0, 0, 0])
    # controller6 = AdversarialMPCController(veh_config, scene_config, cont_config)
    # agent6 = BicycleVehicle(veh_config, scene_config, x0_6, controller6, 6, color='b')
    # sim.addAgent(agent6)

    x0_7 = np.array([-25, -12, 0, 5, 0, 0, 0])
    # x0_5 = np.array([1000, -5, 0, 5, 0, 0, 0])
    controller7 = SafeMPCController(veh_config, scene_config, cont_config)
    agent7 = BicycleVehicle(veh_config, scene_config, x0_7, controller7, 7, color='k')
    # sim.addAgent(agent7)
    
    
    # sim.runSim(end_plot=True, animate=False, save=False, follow_agent_IDs=[None, 4])
    sim.runSim(end_plot=False, animate=True, save=True, follow_agent_IDs=[4,5])
    

    
