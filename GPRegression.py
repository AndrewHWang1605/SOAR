import copy, time
import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
from functools import partial
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

        self.kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # self.GP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        self.GP = MyGPR(kernel=self.kernel, n_restarts_optimizer=9)

        np.random.seed(888)

        
    def importSimData(self):
        sim_success = False
        sim_idx = 0
        while sim_idx < 1:
            sim_idx += 1
            imported_sim_data = importSimDataFromCSV(sim_idx)
            sim_success = imported_sim_data[0]
            if sim_success:
                self.imported_sim_data.append(imported_sim_data)
            # self.imported_sim_data.append(imported_sim_data)



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



    def predictGP(self, data_input):
        prediction = self.GP.predict(data_input, return_std=False)
        return prediction
        

    
    def plotPredictions(self, output, mean_prediction, std_prediction):
        titles = ["ds", "ey", "epsi", "vx", "vy", "omega"]
        normalized_data = np.divide(mean_prediction - output, output)
        plt.figure(0, figsize=(15,8))
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(normalized_data[:,i])
            plt.title(titles[i] + " normalized error")

        plt.figure(1, figsize=(15,8))
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(output[:,i], label="Training data")
            plt.plot(mean_prediction[:,i], label="Mean prediction")
            plt.title(titles[i])

        plt.show()
    


    def getSampleDataVaried(self, count):
        # GP_train_data array will have [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k2]
        GP_train_data = np.zeros((count, 9))
        # GP_output_data array will have full state (7)
        GP_output_data = np.zeros((count, 7))

        len_data = len(self.imported_sim_data)
        if len_data <= 2:
            sim_subcount = count
        else:
            # sim_subcount = int(count/len_data)
            sim_subcount = 2 * count/len_data
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
                if s1 > track_length - self.ds_bound/2 and s2 < self.ds_bound/2:
                    ds -= track_length
                elif s2 > track_length - self.ds_bound/2 and s1 < self.ds_bound/2:
                    ds += track_length

                if abs(ds) <= self.ds_bound:
                    print(total_counter, np.round(ds,2))
                    dey = ey1 - ey2
                    kappa2 = track.getCurvature(s2)
                    GP_train_data[total_counter] = np.array([ds, dey, epsi1, vx1, ey2, epsi2, vx2, omega2, kappa2])
                    
                    """Output data as oppo lookahead state"""
                    # opp_state_p1 = opp_states[sample_idx+self.timestep_offset]
                    # GP_output_data[total_counter] = copy.deepcopy(opp_state_p1)

                    """Output data as oppo lookahead state with ds instead of s2"""
                    ego_state_p1 = ego_states[sample_idx+self.timestep_offset]
                    opp_state_p1 = opp_states[sample_idx+self.timestep_offset]
                    s1, ey1, epsi1, vx1, vy1, omega1, delta1 = ego_state_p1
                    s2, ey2, epsi2, vx2, vy2, omega2, delta2 = opp_state_p1
                    
                    s1 = np.mod(np.mod(s1, track_length) + track_length, track_length)
                    s2 = np.mod(np.mod(s2, track_length) + track_length, track_length)
                    ds = s1 - s2 
                    if s1 > track_length - self.ds_bound/2 and s2 < self.ds_bound/2:
                        ds -= track_length
                    elif s2 > track_length - self.ds_bound/2 and s1 < self.ds_bound/2:
                        ds += track_length

                    GP_output_data[total_counter] = np.array([ds, ey2, epsi2, vx2, vy2, omega2, delta2])

                    counter += 1
                    total_counter += 1
                break_counter += 1
        
        print("\ntotal_counter =", total_counter)
        train_data = GP_train_data[:total_counter-1]
        output_data = GP_output_data[:total_counter-1]
        shuffle_train_data, shuffle_output_data = shuffle(train_data, output_data)
        return shuffle_train_data, shuffle_output_data




# class MyGPR(GaussianProcessRegressor):
#     def __init__(self, kernel, n_restarts_optimizer, max_iter=15000):
#         super().__init__(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
#         self.max_iter = max_iter

#     def _constrained_optimization(self, obj_func, initial_theta, bounds):
#         def new_optimizer(obj_func, initial_theta, bounds):
#             return scipy.optimize.minimize(
#                 obj_func,
#                 initial_theta,
#                 method="L-BFGS-B",
#                 jac=True,
#                 bounds=bounds,
#                 max_iter=self.max_iter,
#             )
#         self.optimizer = new_optimizer
#         return super()._constrained_optimization(obj_func, initial_theta, bounds)


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
    gpr.importSimData()
    gpr.trainGP()
    gpr.testPredictGP(True)