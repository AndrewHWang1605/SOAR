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
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.utils import shuffle
from sklearn.utils.optimize import _check_optimize_result

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

        self.kernel = 1 * Matern(length_scale=1e2, length_scale_bounds=(1e0, 1e5))
        self.GP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        # self.GP = MyGPR(kernel=self.kernel, n_restarts_optimizer=9)

        np.random.seed(888)



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
    def importSimData(self, sim_counts=[1]):
        print("Importing sim data from csv files")
        sim_success = False
        self.imported_sim_data = []
        sim_idx = 0
        for sim_idx in sim_counts:
            imported_sim_data = importSimDataFromCSV(sim_idx)
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
        start_time = time.time()
        mean_prediction, std_prediction = self.GP.predict(random_data, return_std=True)
        end_time = time.time()
        print(np.round(end_time-start_time, 3))
        if end_plot:
            self.plotPredictions(random_output, mean_prediction, std_prediction)



    """Uses fitted GP to predict output for state input(s)"""
    def predict(self, ego_state, opp_states):
        start_time = time.time()
        if len(opp_states.shape) == 1:
            agent_inputs, ds = self.stateToGPInput(ego_state, opp_states, self.track)
        else:
            agent_inputs = np.zeros((9,opp_states.shape[0]))
            for i, opp_state in enumerate(opp_states):
                agent_inputs[i], ds = self.stateToGPInput(ego_state, opp_state, self.track)
        gp_predicts, std_predicts = self.GP.predict(agent_inputs, return_std=True)
        end_time = time.time()
        print("Predict time:", np.round(end_time - start_time, 5))
        return gp_predicts #, std_predicts
        
    

    """Plots prediction vs. outputs and normalized errors"""
    def plotPredictions(self, output, mean_prediction, std_prediction):
        titles = np.array(["ds", "dey", "epsi", "vx", "vy", "omega"])
        elements = output.shape[1]
        titles = titles[:elements]
        normalized_data = np.divide(mean_prediction - output, output)
        counts = np.arange(normalized_data.shape[0])
        plt.figure(0, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.plot(normalized_data[:,i])
            # plt.scatter(counts, normalized_data[:,i])
            plt.title(titles[i] + " normalized error")

        plt.figure(1, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.plot(output[:,i], label="Training data")
            plt.plot(mean_prediction[:,i], label="Mean prediction")
            # plt.scatter(counts, output[:,i], label="Training data")
            # plt.scatter(counts, mean_prediction[:,i], label="Mean prediction")
            plt.title(titles[i])

        plt.show()
    


    """Gets randomly sampled data from multiple imported sims"""
    def getSampleDataVaried(self, count):
        #GP_train_data array will have [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k2]
        GP_train_data = np.zeros((count, 9))
        #GP_output_data array will have ds and de_y for lookahead state
        GP_output_data = np.zeros((count, 2))

        len_data = len(self.imported_sim_data)
        if len_data <= 2:
            sim_subcount = count
        else:
            sim_subcount = count/len_data
        total_counter = 0

        for sim in self.imported_sim_data:
            sim_success, collision_agents, agent_count, track_config, states = sim

            track_type = track_config["track_type"]
            if track_type == 0:
                track = OvalTrack(track_config)
            elif track_type == 1:
                track = LTrack(track_config)

            timesteps = states[0].shape[0]
            ego_idx = 0
            ego_states = np.array(states[ego_idx])

            counter = 0
            break_counter = 0
            while counter < sim_subcount and break_counter < count*self.sample_attempt_repeat and total_counter < count:
                opp_idx = np.random.randint(1, agent_count)
                opp_states = np.array(states[opp_idx])
                sample_idx = np.random.randint(0, timesteps-self.timestep_offset)

                ego_state = ego_states[sample_idx]
                opp_state = opp_states[sample_idx]
                input_data, ds = self.stateToGPInput(ego_state, opp_state, track)

                if abs(ds) <= self.ds_bound:
                    GP_train_data[total_counter] = input_data
                    """Output data as oppo lookahead state with ds instead of s2"""
                    future_oppo_state = opp_states[sample_idx+self.timestep_offset]
                    output_data = self.stateToGPOutput(ego_state, future_oppo_state, track)
                    GP_output_data[total_counter] = output_data

                    print(total_counter, np.round(GP_train_data[total_counter][:2],2), np.round(GP_output_data[total_counter],2))
                    counter += 1
                    total_counter += 1

                break_counter += 1
        
        print("\ntotal_counter =", total_counter)
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
        

    def getDiffS(self, s1, s2, track_length):
        s1 = self.normalizeS(s1, track_length)
        s2 = self.normalizeS(s2, track_length)

        if (s1-s2 > 0.5*track_length):
            return (s1-track_length) - s2
        elif (s1-s2 < -0.5*track_length):
            return s1 - (s2-track_length)
        else:
            return s1-s2

    def normalizeS(self, s, track_length):
        return np.mod(np.mod(s, track_length) + track_length, track_length)
    
    



class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, n_restarts_optimizer, max_iter=2e5, gtol=1e-06):
        super().__init__(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        self.max_iter = max_iter
        self.gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = sp.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self.max_iter, 'gtol': self.gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
    



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

    # file_name = "train_data/CV_test_data/data" + str(dataID) + ".csv"
    # file_name = "train_data/MPC_test_data/data" + str(dataID) + ".csv"
    file_name = "train_data/ADV_test2_data/data" + str(dataID) + ".csv"
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        sim_data = dict(reader)
    csv_file.close()

    sim_success = np.array(ast.literal_eval(sim_data["sim_success"]))
    collision_agents = np.array(ast.literal_eval(sim_data["collision_agents"]))
    # times = np.array(ast.literal_eval(sim_data["t"]))
    agent_count = np.array(ast.literal_eval(sim_data["agent_count"]))
    states, controls = [], []
    for i in range(agent_count):
        states.append(np.array(ast.literal_eval(sim_data["x" + str(i+1)])))
        # controls.append(np.array(ast.literal_eval(sim_data["u" + str(i+1)])))
    track_config = dict(ast.literal_eval(sim_data["track_config"]))
    
    states = np.array(states)
    controls = np.array(controls)

    return sim_success, collision_agents, agent_count, track_config, states, #controls, times








if __name__ == "__main__":
    GP_config = get_GP_config()
    scene_config = get_scene_config()
    gpr = GPRegression(GP_config, scene_config)

    gpr.importSimData()
    # gpr.importSimData(sim_counts=np.arange(4,21))
    gpr.trainGP()
    # gpr.exportGP("gp_models/new/model_5k_300_3-0_ADV.pkl")

    # gpr.importGP("gp_models/new/model_5k_300_3-0_ADV.pkl")
    # gpr.importSimData(sim_counts=np.arange(1,6))
    gpr.testPredict(end_plot=True)
    # ego = np.array([200, -5, 0, 50, 0, 0, 0])
    # opp = np.array([10, -5, 0, 60, 0, 0, 0])
    # gpr.predict(ego, opp)



    """GP setup code to create class instance and import GP object"""
    # GP_config = get_GP_config()
    # scene_config = get_scene_config()
    # gpr = GPRegression(GP_config, scene_config)
    # gpr.importGP("gp_models/model_5k_250_ADV.pkl")

    """GP predict code, input ego state and opponent state(s) as np arrays"""
    # gpr.predict(ego_state, opp_states)