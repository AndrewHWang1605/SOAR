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
from agents import BicycleVehicle
from track import OvalTrack, LTrack
# from config import get_vehicle_config, get_scene_config, get_controller_config
from config import *
import matplotlib.pyplot as plt
from controllers import SinusoidalController, ConstantVelocityController, NominalOptimalController


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

    def runSim(self, end_plot=False):
        print("\nStarting simulation at dt=" + str(self.dt) + " for " + str(self.sim_time) + " seconds")

        retVal = True
        sim_steps = int(self.sim_time / self.dt)
        for i in range(sim_steps):
            # print(i)
            if i%1000 == 0:
                print("Running simulation: ", i, " timesteps passed")

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
                agent.step(agent_states)

        self.t_hist = self.t_hist[:i+2]

        if self.sim_success:
            print("Finished simulation: ", sim_steps, " timesteps passed\n")
        if end_plot:
            self.plotCLStates()
            self.plotAgentTrack()
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

                # Check agent collision with Euclidean distances, add agent IDs to set if collision
                agent_distance = np.linalg.norm(agent_state_global[:2] - oppo_agent_state_global[:2])
                collision_bound = agent.size + oppo_agent.size
                if agent_distance <= collision_bound:
                    collision = True
                    collision_agents.update([agent.ID, oppo_agent.ID])

        return collision, list(collision_agents)
    

    def exportSimData(self):
        # Ensure entire array is printed in string to csv
        np.set_printoptions(threshold=np.inf)

        sim_data = {}

        sim_data["track_config"] = self.scene_config["track_config"]
        sim_data["sim_success"] = self.sim_success
        sim_data["collision_agents"] = np.array2string(np.array(self.collision_agents), separator=',', suppress_small=True)
        sim_data["t"] = np.array2string(self.t_hist, separator=',', suppress_small=True)
        sim_data["agent_count"] = len(self.agents)
        for agent in self.agents:
            sim_data["x" + str(agent.ID)] = np.array2string(agent.getStateHistory(), separator=',', suppress_small=True)
            sim_data["u" + str(agent.ID)] = np.array2string(agent.getControlHistory(), separator=',', suppress_small=True)

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




if __name__ == "__main__":
    """Initialize configurations"""
    veh_config = get_vehicle_config()
    scene_config = get_scene_config(track_type=L_TRACK)
    cont_config = get_controller_config()
     
    sim = Simulator(scene_config)
    
    x0_1 = np.array([0, 0, 0, 50, 0, 0, 0])
    controller1 = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=50)
    agent1 = BicycleVehicle(veh_config, scene_config, x0_1, controller1, 1)
    sim.addAgent(agent1)

    x0_2 = np.array([350, 0, 0, 75, 0, 0, 0])
    controller2 = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=75)
    agent2 = BicycleVehicle(veh_config, scene_config, x0_2, controller2, 2)
    # sim.addAgent(agent2)

    x0_3 = np.array([500, 0, 0, 12, 0, 0, 0])
    controller3 = ConstantVelocityController(veh_config, scene_config, cont_config)
    agent3 = BicycleVehicle(veh_config, scene_config, x0_3, controller3, 3)
    # sim.addAgent(agent3)
    
    sim.runSim(True)
    

    