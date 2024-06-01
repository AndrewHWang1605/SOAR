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

import copy, time, pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.utils import shuffle
from sklearn.utils.optimize import _check_optimize_result


from track import OvalTrack, LTrack
from data_collect_in_out import importSimDataFromCSV
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

        self.timestep_offset = int(self.lookahead/self.dt)

        self.imported_sim_data = []

        self.kernel = 1 * Matern(length_scale=1e2, length_scale_bounds=(1e0, 1e5))
        self.GP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        # self.GP = MyGPR(kernel=self.kernel, n_restarts_optimizer=9)

        np.random.seed(888)

        
    def importSimData(self, sim_counts=np.arange(1,21)):
        print("Importing sim data from csv files")
        sim_success = False
        sim_idx = 0
        for sim_idx in sim_counts:
            imported_sim_data = importSimDataFromCSV(sim_idx)
            sim_success = imported_sim_data[0]
            if sim_success:
                self.imported_sim_data.append(imported_sim_data)
                print("Imported successful sim:", sim_idx)
            else:
                print("Skipped failed sim:", sim_idx)


    def trainGP(self):
        self.train_data, self.output_data = self.getSampleDataVaried(self.sample_count)
        print("Fitting GP to training and output data")
        start_time = time.time()
        self.GP.fit(self.train_data, self.output_data)
        end_time = time.time()
        print(np.round(end_time-start_time, 3))
        print("Finished fitting GP")
        print(self.GP.kernel_)



    def testPredictGP(self, end_plot=False):
        random_data, random_output = self.getSampleDataVaried(self.test_count)
        start_time = time.time()
        mean_prediction, std_prediction = self.GP.predict(random_data, return_std=True)
        end_time = time.time()
        print(np.round(end_time-start_time, 3))

        if end_plot:
            self.plotPredictions(random_output, mean_prediction, std_prediction)


    def exportGP(self, file_path='gp_models/model_base_test.pkl'):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.GP, f)
            print("Exported GP model to pickle file:", file_path)
        except:
            print("Error occurred when exporting GP to pickle file:", file_path)
        

    def importGP(self, file_path='gp_models/model_base_test.pkl'):
        try:
            with open(file_path, 'rb') as f:
                self.GP = pickle.load(f)
            print("Imported GP model from pickle file:", file_path)
        except:
            print("Error occurred when importing GP from pickle file:", file_path)



    def predictGP(self, data_input):
        prediction = self.GP.predict(data_input, return_std=False)
        return prediction
        

    
    def plotPredictions(self, output, mean_prediction, std_prediction):
        titles = np.array(["ds", "ey", "epsi", "vx", "vy", "omega"])
        elements = output.shape[1]
        titles = titles[:elements]
        normalized_data = np.divide(mean_prediction - output, output)
        plt.figure(0, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.plot(normalized_data[:,i])
            plt.title(titles[i] + " normalized error")

        plt.figure(1, figsize=(15,8))
        for i in range(elements):
            plt.subplot(2,2,i+1)
            plt.plot(output[:,i], label="Training data")
            plt.plot(mean_prediction[:,i], label="Mean prediction")
            plt.title(titles[i])

        plt.show()
    


    def getSampleDataVaried(self, count):
        # GP_train_data array will have [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k2]
        GP_train_data = np.zeros((count, 9))
        # GP_output_data array will have ds and de_y for lookahead state
        GP_output_data = np.zeros((count, 2))

        len_data = len(self.imported_sim_data)
        if len_data <= 2:
            sim_subcount = count
        else:
            sim_subcount = count/3
        total_counter = 0

        for sim in self.imported_sim_data:
            sim_success, collision_agents, agent_count, states, controls, times, track_config = sim

            track_type = track_config["track_type"]
            if track_type == 0:
                track = OvalTrack(track_config)
            elif track_type == 1:
                track = LTrack(track_config)
            track_length = track.getTrackLength()

            timesteps = times.shape[0]
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
                s1, ey1, epsi1, vx1, vy1, omega1, delta1 = ego_state
                s2, ey2, epsi2, vx2, vy2, omega2, delta2 = opp_state
                
                s1 = np.mod(np.mod(s1, track_length) + track_length, track_length)
                s2 = np.mod(np.mod(s2, track_length) + track_length, track_length)
                ds = s1 - s2 
                ds = np.mod(np.mod(ds, track_length) + track_length, track_length)
                # if s1 > track_length - self.ds_bound/2 and s2 < self.ds_bound/2:
                #     ds -= track_length
                # elif s2 > track_length - self.ds_bound/2 and s1 < self.ds_bound/2:
                #     ds += track_length

                if abs(ds) <= self.ds_bound:
                    dey = ey1 - ey2
                    kappa2 = track.getCurvature(s2)
                    GP_train_data[total_counter] = np.array([ds, dey, epsi1, vx1, ey2, epsi2, vx2, omega2, kappa2])
                    
                    """Output data as oppo lookahead state"""
                    # opp_state_p1 = opp_states[sample_idx+self.timestep_offset]
                    # GP_output_data[total_counter] = copy.deepcopy(opp_state_p1)

                    """Output data as oppo lookahead state with ds instead of s2"""
                    ego_state_p1 = ego_states[sample_idx+self.timestep_offset]
                    opp_state_p1 = opp_states[sample_idx+self.timestep_offset]
                    s1_p1, ey1_p1, epsi1_p1, vx1_p1, vy1_p1, omega1_p1, delta1_p1 = ego_state_p1
                    s2_p1, ey2_p1, epsi2_p1, vx2_p1, vy2_p1, omega2_p1, delta2_p1 = opp_state_p1
                    
                    s1_p1 = np.mod(np.mod(s1_p1, track_length) + track_length, track_length)
                    s2_p1 = np.mod(np.mod(s2_p1, track_length) + track_length, track_length)
                    ds_p1 = s1_p1 - s2_p1 
                    ds_p1 = np.mod(np.mod(ds_p1, track_length) + track_length, track_length)
                    # if s1_p1 > track_length - self.ds_bound/2 and s2_p1 < self.ds_bound/2:
                    #     ds_p1 -= track_length
                    # elif s2_p1 > track_length - self.ds_bound/2 and s1_p1 < self.ds_bound/2:
                    #     ds_p1 += track_length
                    dey_p1 = ey1_p1 - ey2_p1

                    raw_output_data = np.array([ds_p1, dey_p1, epsi2_p1, vx2_p1, vy2_p1, omega2_p1, delta2_p1])
                    GP_output_data[total_counter] = raw_output_data[:GP_output_data.shape[1]]

                    ds2 = np.mod(np.mod(s2_p1 - s2, track_length) + track_length, track_length)
                    print(total_counter, np.round(ds,2), np.round(ds2, 2))
                    counter += 1
                    total_counter += 1

                break_counter += 1
        
        print("\ntotal_counter =", total_counter)
        train_data = GP_train_data[:total_counter-1]
        output_data = GP_output_data[:total_counter-1]
        shuffle_train_data, shuffle_output_data = shuffle(train_data, output_data)
        return shuffle_train_data, shuffle_output_data


class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, n_restarts_optimizer, max_iter=2e05, gtol=1e-06):
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



if __name__ == "__main__":
    GP_config = get_GP_config()
    scene_config = get_scene_config()
    gpr = GPRegression(GP_config, scene_config)

    # gpr.importSimData(sim_counts=np.arange(1,15))
    # gpr.trainGP()
    # # gpr.exportGP()
    # gpr.exportGP("gp_models/model_5k_500.pkl")

    gpr.importSimData(sim_counts=np.arange(15,21))
    # gpr.importGP()
    gpr.importGP("gp_models/model_5k_500.pkl")
    gpr.testPredictGP(True)