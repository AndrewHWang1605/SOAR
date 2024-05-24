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
            agent_inits = agent_rand_init(agent_count)
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
    cont_config = get_controller_config()
     
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
def agent_rand_init(agent_count):

    agent_inits = np.zeros((agent_count, 7))
    past_starts = []

    for i in range(agent_count):
        new_start = np.random.randint(0, 500)
        while len(past_starts) > 0:
            if np.any(np.abs(np.array(past_starts) - new_start) <= 10):
                new_start = np.random.randint(0, 500)
            else:
                break
        agent_inits[i,0] = new_start
        agent_inits[i,3] = np.random.randint(10, 20)
        past_starts.append(new_start)

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
def importSimData(dataID):
    
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
        states.append(np.array(ast.literal_eval(sim_data["x" + str(i)])))
        controls.append(np.array(ast.literal_eval(sim_data["u" + str(i)])))

    return sim_success, collision_agents, agent_count, states, controls, times







if __name__ == "__main__":

    sim_count = 4
    agent_count = 3
    rand_init = True
    end_plots = False

    generateData(sim_count, agent_count, rand_init, end_plots)


