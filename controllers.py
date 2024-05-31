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
Implement various controllers
"""
import numpy as np
import casadi as ca

STATE_DIM = 7
INPUT_DIM = 2

class Controller:
    def __init__(self, veh_config, scene_config, control_config):
        self.veh_config = veh_config
        self.scene_config = scene_config
        self.control_config = control_config
        self.ctrl_period = None

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        raise NotImplementedError("Inheritance not implemented correctly")

"""Sinusoidal steering with const acceleration (for debugging)"""
class SinusoidalController(Controller):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)
        self.ctrl_period = 1.0 / control_config["sine_ctrl_freq"]

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        w = self.control_config["omega"]
        return 2, w*np.pi/180*np.cos(w*t)

"""Constant velocity controller for trajectory following of track centerline"""
class ConstantVelocityController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, v_ref=12):
        super().__init__(veh_config, scene_config, control_config)
        self.ctrl_period = 1.0 / control_config["pid_ctrl_freq"]
        self.v_ref = v_ref
        self.prev_v_error = 0
        self.total_v_error = 0
        self.delta_prev = 0
        self.prev_theta_error = 0
        self.total_theta_error = 0

        self.delta_error = 0
        self.prev_delta_error = 0
        self.total_delta_error = 0

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """

        """Unpack global config variables and current state variables"""
        accel, steering_rate = 0, 0
        s, ey, epsi = state[:3]
        track = self.scene_config["track"]
        dt = self.scene_config["dt"]
        lf, lr = self.veh_config["lf"], self.veh_config["lr"]
        k_v, k_theta, k_delta = self.control_config["k_v"], self.control_config["k_theta"], self.control_config["k_delta"]

        """Caclulate global position and expected track position/curvature"""
        global_state = track.CLtoGlobal(state)
        x, y, theta, vx, vy, w, delta = global_state
        track_position = track.getTrackPosition(s)
        x_track, y_track, theta_track = track_position
        kappa = track.getCurvature(s)

        """PID control for velocity with acceleration input with anti-windup"""
        v = np.linalg.norm([vx, vy])
        v_error = (self.v_ref - v) 
        accel = (k_v[0] * v_error) + (k_v[1] * self.total_v_error) + (k_v[2] * (v_error - self.prev_v_error))
        if (np.abs(accel) < self.veh_config["max_accel"]):
            self.total_v_error += v_error*dt
        self.prev_v_error = v_error
        accel = np.clip(accel, -self.veh_config["max_accel"], self.veh_config["max_accel"])

        """Pre-calculating heading theta error with overlap check"""
        theta_error = theta_track - theta
        if (theta_track > 13/14*np.pi and theta < 1/14*np.pi):
            theta_error -= 2*np.pi
        elif (theta_track < 1/14*np.pi and theta > 13/14*np.pi):
            theta_error += 2*np.pi

        """Pre-calculating steering delta error with overlap check"""
        beta = np.arcsin(lr * kappa)
        delta_des = np.arctan((lf+lr)/lr * np.tan(beta))
        delta_error = delta_des - delta
        if (delta_des > 13/14*np.pi and delta < 1/14*np.pi):
            delta_error -= 2*np.pi
        elif (delta_des < 1/14*np.pi and delta > 13/14*np.pi):
            delta_error += 2*np.pi
        
        """
        PID control for steering with steering rate input with anti-windup
        Switches between theta error and delta error based on ey
        """
        if (ey > 0 and theta_error < 0) or (ey < 0 and theta_error > 0):
            theta_dot = (k_theta[0] * theta_error) + (k_theta[1] * self.total_theta_error) + (k_theta[2] * (theta_error - self.prev_theta_error))
            if (np.abs(theta_dot) < self.veh_config["max_steer_rate"]):
                self.total_theta_error += theta_error*dt
            self.prev_theta_error = theta_error
            # print("theta_control", theta_dot)
            steering_rate = theta_dot
        else:
            delta_dot = (k_delta[0] * (delta_error)) + (k_delta[1] * self.total_delta_error) + (k_delta[2] * (delta_error - self.prev_delta_error))
            if (np.abs(delta_dot) < self.veh_config["max_steer_rate"]):
                self.total_delta_error += delta_error*dt
            self.prev_delta_error = delta_error
            # print("delta_control", delta_dot)
            steering_rate = delta_dot
        steering_rate = np.clip(steering_rate, -self.veh_config["max_steer_rate"], self.veh_config["max_steer_rate"])

        return accel, steering_rate
    
"""Try to exactly track nominal trajectory (for debugging)"""
class NominalOptimalController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, raceline_file):
        super().__init__(veh_config, scene_config, control_config)
        unpack_file = np.load(raceline_file)
        self.s_hist = unpack_file["s"]
        self.accel_hist = unpack_file["u_a"]
        self.ddelta_hist = unpack_file["u_s"]

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        s, ey, epsi, vx_cl, vy_cl, w, delta = state
        total_len = self.scene_config["track"].total_len
        s = np.mod(np.mod(s, total_len) + total_len, total_len)
        nearest_s_ind = np.where(s >= self.s_hist)[0][-1]
        return self.accel_hist[nearest_s_ind], self.ddelta_hist[nearest_s_ind]

"""MPC to track reference trajectory (based on https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi/blob/master/workshop_github/Python_Implementation/mpc_code.py)"""
class MPCController(Controller):
    def __init__(self,  veh_config, scene_config, control_config, raceline_file):
        super().__init__(veh_config, scene_config, control_config)
        self.ctrl_period = 1.0 / control_config["opt_freq"]
        self.race_line = np.load(raceline_file)
        self.race_line_mat = self.constructRaceLineMat(self.race_line)
        self.mpc_solver, self.solver_args = self.initSolver()
        self.warm_start = {}
        
    def constructRaceLineMat(self, raceline):
        """
        Unpack raceline dictionary object into matrix 
        Returns 10xN (0:time, 1-7: 7 curvilinear states, 8-9: 2 inputs)
        """
        race_line_mat = np.zeros((STATE_DIM+INPUT_DIM+1, self.race_line["t"].shape[0]))
        race_line_mat[0, :] = raceline["t"]
        race_line_mat[1, :] = raceline["s"]
        race_line_mat[2, :] = raceline["e_y"]
        race_line_mat[3, :] = raceline["e_psi"]
        race_line_mat[4, :] = raceline["v_long"]
        race_line_mat[5, :] = raceline["v_tran"]
        race_line_mat[6, :] = raceline["psidot"]
        race_line_mat[7, :] = raceline["delta"]
        race_line_mat[8, :] = raceline["u_a"]
        race_line_mat[9, :] = raceline["u_s"]
        
        return race_line_mat


    def initSolver(self):
        T = self.control_config["T"]        # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                     # Number of discretization steps

        # Construct state cost matrix
        k_s = self.control_config["opt_k_s"]
        k_ey = self.control_config["opt_k_ey"]
        k_epsi = self.control_config["opt_k_epsi"]
        k_vx = self.control_config["opt_k_vx"]
        k_vy = self.control_config["opt_k_vy"]
        k_omega = self.control_config["opt_k_omega"]
        k_delta = self.control_config["opt_k_delta"]
        Q = ca.diagcat(k_s, k_ey, k_epsi, k_vx, k_vy, k_omega, k_delta)

        # Construct input cost matrix
        k_ua = self.control_config["opt_k_ua"]
        k_us = self.control_config["opt_k_us"]
        R = ca.diagcat(k_ua, k_us)

        # State symbolic variables
        s_ca = ca.SX.sym('s')
        ey_ca = ca.SX.sym('ey')
        epsi_ca = ca.SX.sym('epsi')
        vx_ca = ca.SX.sym('vx')
        vy_ca = ca.SX.sym('vy')
        omega_ca = ca.SX.sym('omega')
        delta_ca = ca.SX.sym('delta')
        states = ca.vertcat(
            s_ca,
            ey_ca,
            epsi_ca,
            vx_ca,
            vy_ca,
            omega_ca,
            delta_ca
        )

        # Input symbolic variables
        accel_ca = ca.SX.sym('accel')
        ddelta_ca = ca.SX.sym('ddelta')
        controls = ca.vertcat(
            accel_ca,
            ddelta_ca
        )

        # Reference symbolic variables
        s_ref_ca = ca.SX.sym('s_ref')
        ey_ref_ca = ca.SX.sym('ey_ref')
        epsi_ref_ca = ca.SX.sym('epsi_ref')
        vx_ref_ca = ca.SX.sym('vx_ref')
        vy_ref_ca = ca.SX.sym('vy_ref')
        omega_ref_ca = ca.SX.sym('omega_ref')
        delta_ref_ca = ca.SX.sym('delta_ref')
        kappa_ref_ca = ca.SX.sym('kappa_ref')
        accel_ref_ca = ca.SX.sym('accel_ref')
        ddelta_ref_ca = ca.SX.sym('ddelta_ref')
        reference = ca.vertcat(
            s_ref_ca,
            ey_ref_ca,
            epsi_ref_ca,
            vx_ref_ca,
            vy_ref_ca,
            omega_ref_ca,
            delta_ref_ca,
            kappa_ref_ca,
            accel_ref_ca,
            ddelta_ref_ca
        )

        # Matrix containing all states over all timesteps + 1 [7 x N+1]
        X = ca.SX.sym('X', STATE_DIM, N+1)

        # Matrix containing all control inputs over all timesteps [2 x N]
        U = ca.SX.sym('U', INPUT_DIM, N)

        # Matrix containing initial state, reference states/inputs, and curvature over all timesteps [10 x N]
        # First column is initial state/curvature with zeros for inputs, rest of columns are reference state+curvature+input
        # Column = [s, ey, epsi, vx, vy, omega, delta, kappa, accel, ddelta]
        P = ca.SX.sym('X', STATE_DIM+INPUT_DIM+1, N+1)

        # Define dynamics function
        next_state = self.casadi_dynamics(states, accel_ca, ddelta_ca, kappa_ref_ca)
        f = ca.Function('f', [states, controls, reference], [next_state]) # Maps states, controls, reference (for curvature) to next state

        # Define state constraints
        lbx = ca.DM.zeros((STATE_DIM*(N+1) + INPUT_DIM*N, 1))
        ubx = ca.DM.zeros((STATE_DIM*(N+1) + INPUT_DIM*N, 1))
        lbx[0 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["s"]
        lbx[1 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["ey"]
        lbx[2 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["epsi"]
        lbx[3 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["vx"]
        lbx[4 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["vy"]
        lbx[5 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["omega"]
        lbx[6 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_lb"]["delta"]
        ubx[0 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["s"]
        ubx[1 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["ey"]
        ubx[2 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["epsi"]
        ubx[3 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["vx"]
        ubx[4 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["vy"]
        ubx[5 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["omega"]
        ubx[6 : STATE_DIM*(N+1) : STATE_DIM] = self.control_config["states_ub"]["delta"]

        # Define input constraints
        lbx[STATE_DIM*(N+1) : : INPUT_DIM] = self.control_config["input_lb"]["accel"]
        lbx[STATE_DIM*(N+1) + 1 : : INPUT_DIM] = self.control_config["input_lb"]["ddelta"]
        ubx[STATE_DIM*(N+1) : : INPUT_DIM] = self.control_config["input_ub"]["accel"]
        ubx[STATE_DIM*(N+1) + 1 : : INPUT_DIM] = self.control_config["input_ub"]["ddelta"]

        # Initialize constraints (g) bounds
        MAX_NUM_OPPONENTS = 5
        # lbg = ca.DM.zeros((STATE_DIM*(N+1) + 2*MAX_NUM_OPPONENTS, 1)) # N+1 dynamics constraints, up to 5 opponents (only consider s and ey)
        # ubg = ca.DM.zeros((STATE_DIM*(N+1) + 2*MAX_NUM_OPPONENTS, 1))
        lbg = ca.DM.zeros((STATE_DIM*(N+1), 1)) # N+1 dynamics constraints only, no opponents
        ubg = ca.DM.zeros((STATE_DIM*(N+1), 1))

        # Define cost function
        final_st = X[:,-1]
        cost_fn = (final_st - P[:STATE_DIM,-1]).T @ Q @ (final_st - P[:STATE_DIM,-1]) # Terminal constraint
        g = X[:,0] - P[:STATE_DIM,0] # Set initial state constraint
        for k in range(N):
            st = X[:,k]
            st_next = X[:,k+1]
            con = U[:,k]
            ref = P[:,k]
            cost_fn += (st - P[:STATE_DIM,k]).T @ Q @ (st - P[:STATE_DIM,k]) + \
                       (con - P[-INPUT_DIM:,k]).T @ R @ (con - P[-INPUT_DIM:,k])
            # Define dynamics equality constraint
            g = ca.vertcat(g, st_next - f(st, con, ref))
        
        OPT_variables = ca.vertcat(
            X.reshape((-1,1)),
            U.reshape((-1,1))
        )

        # Configure NLP solver
        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g, 
            'p': P
        }
        opts = {
            'ipopt': {
                'max_iter': 2000,
                'max_wall_time': 1,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }
        solver_args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,  # states/input lower bound
            'ubx': ubx   # states/input upper bound
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver, solver_args
        
        
    # Steps forward dynamics of vehicle one discrete timestep for CasADi symbolic vars
    def casadi_dynamics(self, x, accel, delta_dot, kappa):
        # Expands state variable and precalculates sin/cos
        s, ey, epsi, vx, vy, omega, delta = [x[i] for i in range(STATE_DIM)]
        sin_epsi, cos_epsi = ca.sin(epsi), ca.cos(epsi)
        sin_delta, cos_delta = ca.sin(delta), ca.cos(delta)

        # Expand scene and vehicle config variables
        dt = self.ctrl_period
        m = self.veh_config["m"]
        Iz = self.veh_config["Iz"]
        lf = self.veh_config["lf"]
        lr = self.veh_config["lr"]
        c = self.veh_config["c"]
        rho = 1.225
        Cd = self.veh_config["Cd"]
        SA = self.veh_config["SA"]
        eps = 1e-6 # Avoid divide by 0

        alpha_f = delta - ca.atan2((vy + lf*omega), (vx+eps))
        alpha_r = -ca.atan2((vy - lr*omega), (vx+eps))

        # Calculate various forces 
        Fxf, Fxr = 0, m*accel
        Fyf, Fyr = c*alpha_f, c*alpha_r
        # Fxf, Fxr, Fyf, Fyr = self.saturate_forces(Fxf, Fxr, Fyf, Fyr)
        Fd = 0.5 * rho * SA * Cd * vx**2

        # Calculate x_dot components from dynamics equations
        s_dot = (vx*cos_epsi - vy*sin_epsi) / (1 - ey * kappa)
        ey_dot = vx*sin_epsi + vy*cos_epsi
        epsi_dot = omega - kappa*s_dot
        vx_dot = 1/m * (Fxr - Fd - Fyf*sin_delta + m*vy*omega)
        vy_dot = 1/m * (Fyr + Fyf*cos_delta - m*vx*omega)
        omega_dot = 1/Iz * (lf*Fyf*cos_delta - lr*Fyr)

        # Propogate state variable forwards one timestep with Euler step
        x_dot = ca.vertcat(s_dot, ey_dot, epsi_dot, vx_dot, vy_dot, omega_dot, delta_dot)
        # print("xdot", np.round(x_dot, 4))
        x_new = x + x_dot*dt
        # x_new[6] = np.clip(x_new[6], -self.veh_config["max_steer"], self.veh_config["max_steer"])
        return x_new

    def getRefTrajectory(self, s0, delta_t):
        """
        s: Current longitudinal position
        t: Monotonically increasing vector of time intervals (starting from 0) that we 
           want to evaluate reference trajectory for (eg [0, 0.25, 0.5])
        """
        ref_traj = np.zeros((STATE_DIM+INPUT_DIM, delta_t.shape[0]))

        # Find closest point on reference trajectory and corresponding time
        total_len = self.scene_config["track"].total_len
        s0 = np.mod(np.mod(s0, total_len) + total_len, total_len)
        s_hist = self.race_line_mat[1,:]
        closest_t = np.interp(s0, s_hist, self.race_line_mat[0,:])

        # Shift delta t based on closest current time
        t_hist = closest_t + delta_t

        for i in range(ref_traj.shape[0]):
            ref_traj[i,:] = np.interp(t_hist, self.race_line_mat[0,:], self.race_line_mat[i+1,:])

        return t_hist, ref_traj

    def getWarmStart(self, x_opt, u_opt):
        """ Shifts optimized x, u by one timestep, leaving first column of warm_start_x as zero to fill with true x0 """
        warm_start_x = np.zeros(x_opt.shape)
        warm_start_x[:,1:-1] = x_opt[:,2:] 
        warm_start_x[:,-1] = x_opt[:,-1]

        warm_start_u = np.zeros(u_opt.shape)
        warm_start_u[:,:-1] = u_opt[:,1:]
        warm_start_u[:,-1] = u_opt[:,-1]

        return warm_start_x, warm_start_u


    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        track = self.scene_config["track"]
        T = self.control_config["T"]
        freq = self.control_config["opt_freq"]
        dt = 1.0/freq                    
        delta_t = np.arange(0, T+dt, dt)
        N = delta_t.shape[0]-1
        t_hist, ref_traj = self.getRefTrajectory(state[0], delta_t) # s, ey, epsi, vx, vy, omega, delta, accel, ddelta
        curvature = track.getCurvature(ref_traj[0,:])

        # Initialize params (reference trajectory, curvature)
        # TODO: Add opponent prediction states here
        state_ref = np.hstack((state.reshape((STATE_DIM,1)), ref_traj[:STATE_DIM,1:]))
        # print(state_ref)
        P_mat = np.vstack((state_ref, curvature, ref_traj[STATE_DIM:]))
        self.solver_args['p'] = ca.DM(P_mat)

        # print(P_mat[:,:3])

        # TODO: Initialize warm start 
        if not self.warm_start: # At first iteration, reference is our best warm start
            X0 = ca.DM(ref_traj[:STATE_DIM, :])
            u0 = ca.DM(ref_traj[-INPUT_DIM:, :-1])
        else:
            X0 = ca.DM(self.warm_start["X0"])
            u0 = ca.DM(self.warm_start["u0"])

        self.solver_args['x0'] = ca.vertcat(
            ca.reshape(X0, STATE_DIM*(N+1), 1),
            ca.reshape(u0, INPUT_DIM*N, 1)
        )

        sol = self.mpc_solver(
            x0=self.solver_args['x0'],
            lbx=self.solver_args['lbx'],
            ubx=self.solver_args['ubx'],
            lbg=self.solver_args['lbg'],
            ubg=self.solver_args['ubg'],
            p=self.solver_args['p']
        )

        x_opt = np.array(ca.reshape(sol['x'][: STATE_DIM * (N+1)], STATE_DIM, N+1))
        u_opt = np.array(ca.reshape(sol['x'][STATE_DIM * (N + 1):], INPUT_DIM, N))

        self.warm_start["X0"], self.warm_start["u0"] = self.getWarmStart(x_opt, u_opt)

        # import matplotlib.pyplot as plt 
        # titles = ["s", "ey", "epsi", "vx", "vy", "omega", "delta", "accel", "delta_dot"]
        # plt.figure(0, figsize=(15,8))
        # print(x_opt[:,:3])

        # for i in range(7):
        #     plt.subplot(3,3,i+1)
        #     plt.plot(np.array(x_opt[i,:]).squeeze())
        #     plt.plot(ref_traj[i,:])
        #     plt.title(titles[i])
        # for i in range(7,9):
        #     plt.subplot(3,3,i+1)
        #     plt.plot(u_opt[i-7,:])
        #     plt.title(titles[i])
        #     plt.plot(ref_traj[i,:])
        # plt.show()

        # a = input("Continue? ")
        # if (a == 'n'):
        #     exit()
        
        # exit()
        # s, ey, epsi, vx_cl, vy_cl, w, delta = state
        # total_len = self.scene_config["track"].total_len
        # s = np.mod(np.mod(s, total_len) + total_len, total_len)
        # nearest_s_ind = np.where(s >= self.s_hist)[0][-1]
        # print(nearest_s_ind)
        # return self.accel_hist[nearest_s_ind], self.ddelta_hist[nearest_s_ind]
        return u_opt[0, 0], u_opt[1, 0]

# from config import *
# veh_config = get_vehicle_config()
# scene_config = get_scene_config(track_type=OVAL_TRACK)
# cont_config = get_controller_config(veh_config, scene_config)
# controller = MPCController(veh_config, scene_config, cont_config, "race_lines/oval_raceline.npz")
# controller.getRefTrajectory(3.5, np.linspace(0,0,20))
# controller.computeControl(np.array([300,0,0,0,0,0,0]), [], 0)