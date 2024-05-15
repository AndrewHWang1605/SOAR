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
from track import Track


class Simulator:
    def __init__(self, scen_config):
        self.scen_config = scen_config
        self.agents = []

    @property
    def track(self):
        return self.scen_config["track"]
    
    def addAgent(self, agent):
        self.agents.append(agent)

    def runSim(self):
        sim_time = self.scen_config["sim_time"]
        dt = self.scen_config["dt"]
        sim_steps = int(sim_time / dt)

        for i in range(sim_steps):
            agent_states = {}
            for agent in self.agents:
                agent_states[agent.ID] = agent.getLastState()

            collision, collision_agents = self.checkCollisions()
            if collision:
                print("Collision detected for agents:", *collision_agents)
                return False

            for agent in self.agents:
                agent.step(agent_states)

        return True
    

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
                track = self.scen_config["track"]
                agent_state_global = track.CLToGlobal(agent_state_CL)
                oppo_agent_state_global = track.CLToGlobal(oppo_agent_state_CL)

                # Check agent collision with Euclidean distances, add agent IDs to set if collision
                agent_distance = np.linalg.norm(agent_state_global[:2] - oppo_agent_state_global[:2])
                collision_bound = agent.size + oppo_agent.size
                if agent_distance <= collision_bound:
                    collision = True
                    collision_agents.update([agent.ID, oppo_agent.ID])

        return collision, list(collision_agents)


if __name__ == "__main__":
    pass