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
Implement data collection module
"""

import numpy as np
import csv, ast, sys

from agents import BicycleVehicle
from run_simulation import Simulator
from track import *
from config import *
from controllers import *



"""Runs random simulations and generates data, exporting to csv files"""
def generateRandomData(data_config, control_type, end_plots=False):
    sim_count = data_config["sim_count"]
    agent_count = data_config["agent_count"]
    rand_init = data_config["rand_init"]
    agent_inits = data_config["agent_inits"]
    control_type = control_type

    sims = []

    print("\nRunning simulations to generate data")
    for i in range(sim_count):
        if rand_init:
            agent_init = agentRandomInit(agent_count)
        else:
            agent_init = agent_inits[:agent_count, :]
        
        print("Starting to run simulation #", i+1)
        sim = runSimulation(agent_init, control_type, end_plots)
        exportSimDataToCSV(sim, i+1)
        print("Finished exporting simulation #", i+1)
        sims.append(sim)
        print("Finished running simulation #", i+1, "\n")



"""Runs uniform simulations and generates data, exporting to csv files"""
def generateUniformData(data_config, control_type, end_plots=False):
    agent_count = data_config["agent_count"]
    control_type = control_type

    agent_inits, sim_count = agentUniformInit(agent_count)
    sims = []

    print("\nRunning simulations to generate data")
    for i in range(sim_count):
        agent_init = agent_inits[i, :agent_count, :]
        print(agent_init)
        
        print("Starting to run simulation #", i+1)
        sim = runSimulation(agent_init, control_type, end_plots)
        exportSimDataToCSV(sim, i+1)
        print("Finished exporting simulation #", i+1)
        sims.append(sim)
        print("Finished running simulation #", i+1, "\n")



"""Initializes a new simulator with agent init states and runs it"""
def runSimulation(agent_inits, control_type, end_plots=False):
    """Initialize configurations"""
    veh_config = get_vehicle_config()
    scene_config = get_scene_config(track_type=L_TRACK)
    cont_config = get_controller_config(veh_config, scene_config)
     
    sim = Simulator(scene_config)

    for i in range(len(agent_inits)):
        x0 = np.array(agent_inits[i])
        if control_type[i] == ConstantVelocityController:
            v_ref = np.linalg.norm(x0[3:5])
            controller = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=v_ref)
        elif control_type[i] == MPCController:
            controller = MPCController(veh_config, scene_config, cont_config)
        elif control_type[i] == AdversarialMPCController:
            controller = MPCController(veh_config, scene_config, cont_config)
        agent_ID = i+1
        agent = BicycleVehicle(veh_config, scene_config, x0, controller, agent_ID)
        sim.addAgent(agent)

    sim.runSim(end_plots)
    return sim



"""Generates random agent initial states (only s and vx for now in CL frame)"""
def agentRandomInit(agent_count):

    agent_inits = np.zeros((agent_count, 7))
    past_starts = []

    new_start_ref = np.random.randint(200,10000)
    for i in range(agent_count):
        new_start = np.random.randint(new_start_ref-100, new_start_ref+100)
        while len(past_starts) > 0:
            if np.any(np.abs(np.array(past_starts) - new_start) <= 50):
                new_start = np.random.randint(new_start_ref-100, new_start_ref+100)
            else:
                break
        agent_inits[i,0] = new_start
        agent_inits[i,1] = np.random.randint(-5, 5)
        agent_inits[i,3] = np.random.randint(0, 20)
        past_starts.append(new_start)
        print("Agent", i, "initialized as:", agent_inits[i,0], agent_inits[i,3])

    return agent_inits



def agentUniformInit(agent_count):

    start_s = np.arange(100,4800,800)
    start_vel = np.arange(5,105,5)
    sim_count = len(start_s)*len(start_vel)

    agent_inits = np.zeros((sim_count, agent_count, 7))

    counter = 0
    for i, s in enumerate(start_s):
        for j, v in enumerate(start_vel):
            agent_inits[counter, 0, 0] = s
            agent_inits[counter, 0, 1] = np.random.randint(-15, 15)
            agent_inits[counter, 0, 3] = v + np.random.randint(-3, 3)
            past_starts = [s]
            for k in range(1,agent_count):
                new_start = np.random.randint(s-75, s+75)
                while len(past_starts) > 0:
                    if np.any(np.abs(np.array(past_starts) - new_start) <= 25):
                        new_start = np.random.randint(s-75, s+75)
                    else:
                        break
                agent_inits[counter, k, 0] = new_start
                agent_inits[counter, k, 1] = np.random.randint(-15, 15)
                agent_inits[counter, k, 3] = v + np.random.randint(-3, 3)
                past_starts.append(new_start)
            counter += 1
    
    return agent_inits, sim_count
            


"""Exports a sim's data to a csv file"""
def exportSimDataToCSV(sim, dataID):
    sim_data = sim.exportSimData()
    file_name = "train_data/ADV_handicap_data/data" + str(dataID) + ".csv"

    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sim_data.items():
            writer.writerow([key, value])
    csv_file.close()



if __name__ == "__main__":

    data_config = get_data_collect_config()
    control_type = [MPCController, AdversarialMPCController]
    # generateRandomData(data_config, control_type, end_plots=False)
    generateUniformData(data_config, control_type, end_plots=False)


