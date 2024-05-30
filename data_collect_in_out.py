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
import pandas as pd

from agents import BicycleVehicle
from config import get_vehicle_config, get_scene_config, get_controller_config
from controllers import ConstantVelocityController
from run_simulation import Simulator


"""Runs simulations and generates data, exporting to csv files"""
def generateData(sim_count, agent_count, rand_init=False, end_plots=False):
    agent_inits = np.array([[260, 0, 0, 15, 0, 0, 0],
                            [240, 0, 0, 12, 0, 0, 0],
                            [500, 0, 0, 12, 0, 0, 0]])
    sims = []

    print("\nRunning simulations to generate data")
    for i in range(sim_count):
        if rand_init:
            agent_inits = agentRandomInit(agent_count)
        else:
            agent_inits = agent_inits[:agent_count, :]
        
        print("Starting to run simulation #", i+1)
        sim = runSimulation(agent_inits, end_plots)
        sims.append(sim)
        print("Finished running simulation #", i+1, "\n")

    print("Exporting simulation data to csv files")
    for i in range(sim_count):
        exportSimDataToCSV(sims[i], i+1)
        print("Finished exporting simulation #", i+1)



"""Initializes a new simulator with agent init states and runs it"""
def runSimulation(agent_inits, end_plots=False):
    """Initialize configurations"""
    veh_config = get_vehicle_config()
    scene_config = get_scene_config()
    cont_config = get_controller_config(veh_config, scene_config)
     
    sim = Simulator(scene_config)

    for i in range(len(agent_inits)):
        x0 = np.array(agent_inits[i])
        v_ref = np.linalg.norm(x0[3:5])
        controller = ConstantVelocityController(veh_config, scene_config, cont_config, v_ref=v_ref)
        agent = BicycleVehicle(veh_config, scene_config, x0, controller, i+1)
        sim.addAgent(agent)

    sim.runSim(end_plots)
    return sim



"""Generates random agent initial states (only s and vx for now in CL frame)"""
def agentRandomInit(agent_count):

    agent_inits = np.zeros((agent_count, 7))
    past_starts = []

    new_start_ref = np.random.randint(0,3000)
    for i in range(agent_count):
        new_start = np.random.randint(new_start_ref, new_start_ref+400)
        while len(past_starts) > 0:
            if np.any(np.abs(np.array(past_starts) - new_start) <= 100):
                new_start = np.random.randint(new_start_ref, new_start_ref+400)
            else:
                break
        agent_inits[i,0] = new_start
        agent_inits[i,3] = np.random.randint(60, 80)
        past_starts.append(new_start)
        print("Agent", i, "initialized as:", agent_inits[i,0], agent_inits[i,3])

    return agent_inits



"""Exports a sim's data to a csv file"""
def exportSimDataToCSV(sim, dataID):
    sim_data = sim.exportSimData()
    file_name = "train_data/data" + str(dataID) + ".csv"

    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sim_data.items():
            writer.writerow([key, value])
    csv_file.close()



"""Imports a sim's data from a csv file"""
def importSimDataFromCSV(dataID):
    
    # Decrease the maxInt value by factor 10 if OverflowError for dict import
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    file_name = "train_data/data" + str(dataID) + ".csv"
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        sim_data = dict(reader)
    csv_file.close()

    sim_success = np.array(ast.literal_eval(sim_data["sim_success"]))
    collision_agents = np.array(ast.literal_eval(sim_data["collision_agents"]))
    times = np.array(ast.literal_eval(sim_data["t"]))
    agent_count = np.array(ast.literal_eval(sim_data["agent_count"]))
    states, controls = [], []
    for i in range(agent_count):
        states.append(np.array(ast.literal_eval(sim_data["x" + str(i+1)])))
        controls.append(np.array(ast.literal_eval(sim_data["u" + str(i+1)])))
    track_config = dict(ast.literal_eval(sim_data["track_config"]))

    return sim_success, collision_agents, agent_count, states, controls, times, track_config







if __name__ == "__main__":

    sim_count = 20
    agent_count = 2
    rand_init = True
    end_plots = False

    generateData(sim_count, agent_count, rand_init, end_plots)


