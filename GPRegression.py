import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern


from GPStructure import GaussianProcess, radial_basis, matern
from track import OvalTrack, LTrack
from data_collect_in_out import importSimDataFromCSV
from config import get_GP_config




class GaussianProcessRegression():

    def __init__(self, config):
        self.config = config
        self.sample_count = config["sample_count"]
        self.test_count = config["test_count"]
        self.ds_bound = config["ds_bound"]

        self.imported_sim_data = []

        self.kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.GP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)

        np.random.seed(888)

        


    def importSimData(self):
        sim_success = False
        sim_idx = 0
        while sim_idx < 4:
            sim_idx += 1
            imported_sim_data = importSimDataFromCSV(sim_idx)
            sim_success = imported_sim_data[0]
            if sim_success:
                self.imported_sim_data.append(imported_sim_data)



    def trainGP(self):
        self.GP_train_data, self.GP_output_data = self.getSampleData(self.sample_count)
        print("Fitting GP to training and output data")
        self.GP.fit(self.GP_train_data, self.GP_output_data)
        print("Finished fitting GP")
        print(self.GP.kernel_)


    def predictGP(self):
        random_data, random_output = self.getSampleData(self.test_count)
        mean_prediction, std_prediction = self.GP.predict(random_data, return_std=True)
        plt.plot(random_output[:,0], label="Training data")
        plt.plot(mean_prediction[:,0], label="Mean prediction")
        plt.show()



    def getSampleData(self, count):
        sim_success, collision_agents, agent_count, states, controls, times, track_config = self.imported_sim_data[0]

        track_type = track_config["track_type"]
        if track_type == 0:
            track = OvalTrack(track_config)
        elif track_type == 1:
            track = LTrack(track_config)
        track_length = track.getTrackLength()

        timesteps = times.shape[0]
        opp_count = agent_count - 1
        ego_idx = 0
        ego_states = np.array(states[ego_idx])
        print(times.shape)
        print(ego_states.shape)
    
        # GP_train_data array will have [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k2]
        GP_train_data = np.zeros((count, 9))
        # GP_output_data array will have full state (7)
        GP_output_data = np.zeros((count, 7))

        for opp_idx in range(opp_count):
            opp_states = np.array(states[opp_idx])
            counter = 0
            while counter < count:
                
                sample_idx = np.random.randint(0, timesteps-50)

                ego_state = ego_states[sample_idx]
                opp_state = opp_states[sample_idx]
                s1, ey1, epsi1, vx1, vy1, omega1, delta1 = ego_state
                s2, ey2, epsi2, vx2, vy2, omega2, delta2 = opp_state
                
                s1 = np.mod(np.mod(s1, track_length) + track_length, track_length)
                s2 = np.mod(np.mod(s2, track_length) + track_length, track_length)
                ds = s1 - s2
                print(counter, ds)
                if abs(ds) <= self.ds_bound:
                    dey = ey1 - ey2
                    kappa2 = track.getCurvature(s2)
                    GP_train_data[counter] = np.array([ds, dey, epsi1, vx1, ey2, epsi2, vx2, omega2, kappa2])
                    opp_state_p1 = opp_states[sample_idx+50]
                    GP_output_data[counter] = copy.deepcopy(opp_state_p1)
                    counter += 1
        
        return GP_train_data, GP_output_data



if __name__ == "__main__":
    GP_config = get_GP_config()
    gpr = GaussianProcessRegression(GP_config)
    gpr.importSimData()
    gpr.trainGP()
    gpr.predictGP()