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
Implement Gaussian process module
"""

import csv, ast, sys, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import shuffle

from track import OvalTrack, LTrack
from config import *


class GPRegression():

    def __init__(self, GP_config, scene_config):
        self.GP_config = GP_config
        self.scene_config = scene_config

        self.sample_count = GP_config["sample_count"]
        self.sample_attempt_repeat = GP_config["sample_attempt_repeat"]
        self.test_count = GP_config["test_count"]
        self.ds_bound = GP_config["ds_bound"]
        self.lookahead = GP_config["lookahead"]
        self.dt = scene_config["dt"]
        self.track = self.scene_config["track"]

        self.timestep_offset = int(self.lookahead/self.dt)

        self.imported_sim_data = []

        self.kernel = 1 * Matern(length_scale=1e2, length_scale_bounds=(1e1, 1e6))
        self.GP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)


    """Exports current GP object to pickle"""
    def exportGP(self, file_path='gp_models/model_base_test.pkl'):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.GP, f)
            print("Exported GP model to pickle file:", file_path)
        except:
            print("Error occurred when exporting GP to pickle file:", file_path)
        

    """Imports GP object from saved pickle"""
    def importGP(self, file_path='gp_models/model_base_test.pkl'):
        try:
            with open(file_path, 'rb') as f:
                self.GP = pickle.load(f)
            print("Imported GP model from pickle file:", file_path)
        except:
            print("Error occurred when importing GP from pickle file:", file_path)


    """Imports sim data from filepath, can select multiple sims"""
    def importSimData(self, file_path, sim_counts=[1]):
        print("Importing sim data from csv files")
        sim_success = False
        self.imported_sim_data = []
        sim_idx = 0
        for sim_idx in sim_counts:
            imported_sim_data = importSimDataFromCSV(sim_idx, file_path)
            sim_success = imported_sim_data[0]
            if sim_success:
                self.imported_sim_data.append(imported_sim_data)
                print("Imported successful sim:", sim_idx)
            else:
                print("Skipped collision sim:", sim_idx)


    """Trains GP by fitting input/outputs from inputted sim data"""
    def trainGP(self):
        self.train_data, self.output_data = self.getSampleDataVaried(self.sample_count)
        print("Fitting GP to training and output data")
        start_time = time.time()
        self.GP.fit(self.train_data, self.output_data)
        end_time = time.time()
        print(np.round(end_time-start_time, 3))
        print("Finished fitting GP")
        print(self.GP.kernel_)


    """Tests GP prediction from inputted sim data"""
    def testPredict(self, end_plot=False):
        random_data, random_output = self.getSampleDataVaried(self.test_count)
        mean_prediction, std_prediction = self.GP.predict(random_data, return_std=True)
        if end_plot:
            self.plotPredictions(random_output, mean_prediction, std_prediction)


    """Uses fitted GP to predict output for state input(s)"""
    def predict(self, ego_state, opp_states):
        if len(opp_states.shape) == 1:
            agent_inputs, ds = self.stateToGPInput(ego_state, opp_states, self.track)
        else:
            agent_inputs = np.zeros((9,opp_states.shape[0]))
            for i, opp_state in enumerate(opp_states):
                agent_inputs[i], ds = self.stateToGPInput(ego_state, opp_state, self.track)
        gp_predicts, std_predicts = self.GP.predict(agent_inputs, return_std=True)
        return gp_predicts
        

    """Plots predictions vs. actual outputs and normalized errors"""
    def plotPredictions(self, output, mean_prediction, std_prediction):
        titles = np.array(["ds", "dey"])
        elements = output.shape[1]
        titles = titles[:elements]
        counts = np.arange(mean_prediction.shape[0])

        plt.figure(0, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.scatter(counts, output[:,i], label="Training data", marker=".", s=25)
            plt.scatter(counts, mean_prediction[:,i], label="Mean prediction", marker=".", s=25)
            plt.title(titles[i] + "prediction vs. actual")

        mean_prediction[:,1] = mean_prediction[:,1] + self.scene_config["track_config"]["track_half_width"]
        output[:,1] = output[:,1] + self.scene_config["track_config"]["track_half_width"]
        normalized_data = np.abs(np.divide(mean_prediction - output, output))

        plt.figure(1, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.scatter(counts, normalized_data[:,i], marker=".", s=25)
            plt.title(titles[i] + " Normalized Error")

        print("Average normalized errors:", np.mean(normalized_data,axis=0))
        plt.show()


    """Gets randomly sampled data from multiple imported sims"""
    def getSampleDataVaried(self, count):
        # [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k2]
        GP_train_data = np.zeros((count, 9))
        # [ds_f and de_y_f]
        GP_output_data = np.zeros((count, 2))

        # Determine how many datapoints sampled per sim
        len_data = len(self.imported_sim_data)
        if len_data <= 2:
            sim_subcount = count
        else:
            sim_subcount = count/len_data

        # Cycle through sims to sample datapoints
        total_counter = 0
        for sim in self.imported_sim_data:
            # Export sim scene and agent information
            sim_success, collision_agents, agent_count, track_config, states = sim

            track_type = track_config["track_type"]
            if track_type == OVAL_TRACK:
                track = OvalTrack(track_config)
            elif track_type == L_TRACK:
                track = LTrack(track_config)

            timesteps = states[0].shape[0]
            ego_idx = 0
            ego_states = np.array(states[ego_idx])

            # Sample datapoints from sim that are within ds bound
            current_counter = 0
            break_counter = 0
            while (current_counter < sim_subcount) and (break_counter < count*self.sample_attempt_repeat) and (total_counter < count):
                opp_idx = np.random.randint(1, agent_count)
                opp_states = np.array(states[opp_idx])
                sample_idx = np.random.randint(0, timesteps-self.timestep_offset)

                ego_state = ego_states[sample_idx]
                opp_state = opp_states[sample_idx]
                input_data, ds = self.stateToGPInput(ego_state, opp_state, track)

                if abs(ds) <= self.ds_bound:
                    GP_train_data[total_counter] = input_data
                    future_oppo_state = opp_states[sample_idx+self.timestep_offset]
                    output_data = self.stateToGPOutput(ego_state, future_oppo_state, track)
                    GP_output_data[total_counter] = output_data

                    current_counter += 1
                    total_counter += 1

                break_counter += 1

            print("Collected datapoints:", total_counter)
        
        # Trim empty array elements from data array, shuffle, and return
        print("\nTotal collected datapoints:", total_counter)
        train_data = GP_train_data[:total_counter-1]
        output_data = GP_output_data[:total_counter-1]
        shuffle_train_data, shuffle_output_data = shuffle(train_data, output_data)
        return shuffle_train_data, shuffle_output_data


    """Helper function that returns the GP input"""
    def stateToGPInput(self, ego_state, opp_state, track):
        track_length = track.getTrackLength()
        s1, ey1, epsi1, vx1, vy1, omega1, delta1 = ego_state
        s2, ey2, epsi2, vx2, vy2, omega2, delta2 = opp_state
        
        ds = self.getDiffS(s1, s2, track_length)
        dey = ey1 - ey2
        kappa2 = track.getCurvature(s2)

        predict_input = np.array([ds, dey, epsi1, vx1, ey2, epsi2, vx2, omega2, kappa2])
        predict_input = np.reshape(predict_input, (1, predict_input.shape[0]))
        return predict_input, ds
    

    """Helper function that returns the GP output"""
    def stateToGPOutput(self, ego_state, future_opp_state, track):
        track_length = track.getTrackLength()
        s1, ey1, epsi1, vx1, vy1, omega1, delta1 = ego_state
        s2, ey2, epsi2, vx2, vy2, omega2, delta2 = future_opp_state
        
        ds = self.getDiffS(s1, s2, track_length)
        dey = ey1 - ey2

        predict_output = np.array([ds, dey])
        predict_output = np.reshape(predict_output, (1, predict_output.shape[0]))
        return predict_output
        

    """Helper function to get s differential with track wrap-around"""
    def getDiffS(self, s1, s2, track_length):
        s1 = self.normalizeS(s1, track_length)
        s2 = self.normalizeS(s2, track_length)

        if (s1-s2 > 0.5*track_length):
            return (s1-track_length) - s2
        elif (s1-s2 < -0.5*track_length):
            return s1 - (s2-track_length)
        else:
            return s1-s2


    """Helper function to normalize s by track length"""
    def normalizeS(self, s, track_length):
        return np.mod(np.mod(s, track_length) + track_length, track_length)
    



"""Imports a sim's data from a csv file"""
def importSimDataFromCSV(dataID, file_path):
    # Decrease the maxInt value by factor 10 if OverflowError for dict import
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    file_name = file_path + "/data" + str(dataID) + ".csv"
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        sim_data = dict(reader)
    csv_file.close()

    sim_success = np.array(ast.literal_eval(sim_data["sim_success"]))
    collision_agents = np.array(ast.literal_eval(sim_data["collision_agents"]))
    agent_count = np.array(ast.literal_eval(sim_data["agent_count"]))
    states = []
    for i in range(agent_count):
        states.append(np.array(ast.literal_eval(sim_data["x" + str(i+1)])))
    track_config = dict(ast.literal_eval(sim_data["track_config"]))
    
    states = np.array(states)
    return sim_success, collision_agents, agent_count, track_config, states




if __name__ == "__main__":
    GP_config = get_GP_config()
    scene_config = get_scene_config()
    gpr = GPRegression(GP_config, scene_config)
    file_path = "train_data/ADV_handicap_data"

    if GP_config["training"]:
        sim_counts = np.arange(1,241)
        gpr.importSimData(file_path, sim_counts)
        gpr.trainGP()
        gpr.exportGP("gp_models/ADV_handicap/test.pkl")

    if GP_config["testing"]:
        gpr.importGP("gp_models/ADV_handicap/model_5k_250_3-0_ADV.pkl")
        sim_counts = np.random.randint(1,241,60)
        gpr.importSimData(file_path, sim_counts)
        gpr.testPredict(end_plot=True)