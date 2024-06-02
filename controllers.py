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
import time, copy

from GPRegression import GPRegression

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
        if (state[3] < self.control_config["jumpstart_velo"]): # Handles weirdness at very low speeds (accelerates to small velo, then controller kicks in)
            return self.control_config["input_ub"]["accel"], 0
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
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)
        unpack_file = np.load(control_config["raceline_filepath"])
        self.s_hist = unpack_file["s"]
        self.accel_hist = unpack_file["u_a"]
        self.ddelta_hist = unpack_file["u_s"]

    def computeControl(self, state, oppo_states, t):
        """
        Calculate next input (rear wheel commanded acceleration, derivative of steering angle) 
        """
        if (state[3] < self.control_config["jumpstart_velo"]): # Handles weirdness at very low speeds (accelerates to small velo, then controller kicks in)
            return self.control_config["input_ub"]["accel"], 0
        s, ey, epsi, vx_cl, vy_cl, w, delta = state
        total_len = self.scene_config["track"].total_len
        s = self.scene_config["track"].normalizeS(s)
        nearest_s_ind = np.where(s >= self.s_hist)[0][-1]
        return self.accel_hist[nearest_s_ind], self.ddelta_hist[nearest_s_ind]


"""MPC to track reference trajectory (based on https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi/blob/master/workshop_github/Python_Implementation/mpc_code.py)"""
class MPCController(Controller):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)
        self.ctrl_period = 1.0 / control_config["opt_freq"]
        self.race_line = np.load(control_config["raceline_filepath"])
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
        """ Initialize CasADi problem and solver for use at each iteration. Generalized for all MPC variants """
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

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

        # Matrix containing all states over all timesteps + 1 [7 x N+1]
        X = ca.SX.sym('X', STATE_DIM, N+1)

        # Matrix containing all control inputs over all timesteps [2 x N]
        U = ca.SX.sym('U', INPUT_DIM, N)

        # Initialize parameter matrix
        P, reference = self.initPmatrix()

        # Define dynamics function
        next_state, force_norm = self.casadiDynamics(states, accel_ca, ddelta_ca, reference[STATE_DIM])
        f = ca.Function('f', [states, controls, reference], [next_state]) # Maps states, controls, reference (for curvature) to next state
        force_fun = ca.Function('force_fun', [states, controls, reference], [force_norm])

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
        lbg, ubg = self.configureConstraints() # KEY: Assumes that lbg/ubg generated here matches order of g generated below

        # Define cost function
        final_st = X[:,-1]
        cost_fn = self.terminalCostFn(final_st, P[:,N]) # Terminal cost
        g_custom = ca.DM()
        g_custom = self.updateTerminalCustomConstraints(g_custom, final_st, P[:,N]) # Custom terminal constraints
        g = X[:,0] - P[:STATE_DIM,0] # Set initial state constraint
        g_maxforce = ca.DM()
        for k in range(N):
            st = X[:,k]
            st_next = X[:,k+1]
            con = U[:,k]
            ref = P[:,k]
            cost_fn += self.stageCostFn(st, con, ref)
            # Define dynamics equality constraint
            g = ca.vertcat(g, st_next - f(st, con, ref))
            g_maxforce = ca.vertcat(g_maxforce, force_fun(st, con, ref))
            g_custom = self.updateStageCustomConstraints(g_custom, st, con, ref)
        g = ca.vertcat(g, g_maxforce, g_custom)
        
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
                'max_wall_time': 1, #s
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

    def stageCostFn(self, st, con, ref, opp=None):  
        """ Define stage cost (penalize state and input deviations from reference) """
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

        k_ddelta = self.control_config["opt_k_ddelta"]

        return (st - ref[:STATE_DIM]).T @ Q @ (st - ref[:STATE_DIM]) + \
               (con - ref[-INPUT_DIM:]).T @ R @ (con - ref[-INPUT_DIM:]) + \
               con[-1].T @ k_ddelta @ con[-1]
               

    def terminalCostFn(self, final_st, ref, opp=None): 
        """ Define terminal cost (penalize deviation from terminal state in reference trajectory), 
            currently the same cost weight as every other time step  """ 
        # Construct state cost matrix
        k_s = self.control_config["opt_k_s"]
        k_ey = self.control_config["opt_k_ey"]
        k_epsi = self.control_config["opt_k_epsi"]
        k_vx = self.control_config["opt_k_vx"]
        k_vy = self.control_config["opt_k_vy"]
        k_omega = self.control_config["opt_k_omega"]
        k_delta = self.control_config["opt_k_delta"]
        Q = ca.diagcat(k_s, k_ey, k_epsi, k_vx, k_vy, k_omega, k_delta)

        return (final_st - ref[:STATE_DIM]).T @ Q @ (final_st - ref[:STATE_DIM])

    def configureConstraints(self):
        """ Configure constraints upper and lower bounds (assumes consistent with updateStageCustomConstraints) """
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

        lbg = ca.DM.zeros((STATE_DIM*(N+1) + (N), 1)) # N+1 dynamics constraints, N force constraints
        ubg = ca.DM.zeros((STATE_DIM*(N+1) + (N), 1))
        ubg[-(N):] = self.veh_config["downforce_coeff"] * self.veh_config["m"] * 9.81 # Max force constraint

        return lbg, ubg

    def updateStageCustomConstraints(self, g_custom, st, con, ref):
        """ Placeholder for subclasses to add additional stagewise constraints """
        return g_custom

    def updateTerminalCustomConstraints(self, g_custom, final_st, ref):
        """ Placeholder for subclasses to add additional terminal constraints """
        return g_custom

    def initPmatrix(self):
        """ 
        Initialize Matrix containing initial state, reference states/inputs, and curvature over all timesteps [10 x N]
        First column is initial state/curvature with zeros for inputs, rest of columns are reference state+curvature+input
        Column = [s, ey, epsi, vx, vy, omega, delta, kappa, accel, ddelta]
        """
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

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

        return ca.SX.sym('P', STATE_DIM+INPUT_DIM+1, N+1), reference

    def constructPmatrix(self, state, ref_traj, oppo_states):
        """ Initialize params (reference trajectory states/inputs, curvature) """
        track = self.scene_config["track"]
        curvature = track.getCurvature(ref_traj[0,:])

        state_ref = np.hstack((state.reshape((STATE_DIM,1)), ref_traj[:STATE_DIM,1:]))
        P_mat = np.vstack((state_ref, curvature, ref_traj[STATE_DIM:]))
        return P_mat

    def casadiDynamics(self, x, accel, delta_dot, kappa):
        """ Steps forward dynamics of vehicle one discrete timestep for CasADi symbolic vars """
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
        x_new = x + x_dot*dt
        force_norm = ca.norm_2(ca.vertcat(Fxr, Fyf, Fyr)) # Assume Fxf = 0
        return x_new, force_norm

    def getRefTrajectory(self, s0, delta_t):
        """
        s0: Current longitudinal position
        delta_t: Monotonically increasing vector of time intervals (starting from 0) that we 
           want to evaluate reference trajectory for (eg [0, 0.25, 0.5])
        """
        ref_traj = np.zeros((STATE_DIM+INPUT_DIM, delta_t.shape[0]))

        # Find closest point on reference trajectory and corresponding time
        track = self.scene_config["track"]
        s0_mult = np.floor_divide(s0, track.total_len) # Number of laps already completed
        s0 = track.normalizeS(s0)
        s_hist = self.race_line_mat[1,:]
        closest_t = np.interp(s0, s_hist, self.race_line_mat[0,:])

        # Shift delta t based on closest current time
        t_hist = closest_t + delta_t
        t_hist_mult = np.floor_divide(t_hist, self.race_line_mat[0, -1]) # Indicates whether lap completed in the middle of horizon
        t_hist = np.mod(t_hist, self.race_line_mat[0, -1])
        
        for i in range(ref_traj.shape[0]):
            ref_traj[i,:] = np.interp(t_hist, self.race_line_mat[0,:], self.race_line_mat[i+1,:])
        ref_traj[0,:] += track.total_len * (t_hist_mult + s0_mult) # Add lap multiples back, so s monotonically increases

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
        t = time.time()
        if (state[3] < self.control_config["jumpstart_velo"]): # Handles weirdness at very low speeds (accelerates to small velo, then controller kicks in)
            return self.control_config["input_ub"]["accel"], 0
        T = self.control_config["T"]
        freq = self.control_config["opt_freq"]
        dt = 1.0/freq                    
        delta_t = np.arange(0, T+dt, dt)
        N = int(T*freq)
        t_hist, ref_traj = self.getRefTrajectory(state[0], delta_t) # s, ey, epsi, vx, vy, omega, delta, accel, ddelta

        # Initialize params
        P_mat = self.constructPmatrix(state, ref_traj, oppo_states)
        self.solver_args['p'] = ca.DM(P_mat)

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

        # self.lookUnderTheHood(x_opt, u_opt, ref_traj)
        if not self.mpc_solver.stats()["success"]:
            print("=== FAILED:", self.mpc_solver.stats()["return_status"])
        print("Compute Time", time.time()-t)
        return u_opt[0, 0], u_opt[1, 0]

    def lookUnderTheHood(self, x_opt, u_opt, ref_traj):
        """ Plot optimized trajectory compared to reference, for debugging """
        import matplotlib.pyplot as plt 
        titles = ["s", "ey", "epsi", "vx", "vy", "omega", "delta", "accel", "delta_dot"]
        plt.figure(0, figsize=(15,8))
        print(x_opt[:,:3])

        for i in range(7):
            plt.subplot(3,3,i+1)
            plt.plot(np.array(x_opt[i,:]).squeeze())
            plt.plot(ref_traj[i,:])
            plt.title(titles[i])
        for i in range(7,9):
            plt.subplot(3,3,i+1)
            plt.plot(u_opt[i-7,:])
            plt.title(titles[i])
            plt.plot(ref_traj[i,:])
        plt.show()

        a = input("Continue? ")
        if (a == 'n'):
            exit()

class AdversarialMPCController(MPCController):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)

    def initPmatrix(self):
        """
        Matrix containing initial state, reference states/inputs, curvature, and nearest opponent ey position over all timesteps [11 x N+1]
        First column is initial state/curvature with zeros for inputs, rest of columns are reference state+curvature+input
        Column = [s, ey, epsi, vx, vy, omega, delta, kappa, *opponent ey*, accel, ddelta]
        """
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

        # Reference symbolic variables
        s_ref_ca = ca.SX.sym('s_ref')
        ey_ref_ca = ca.SX.sym('ey_ref')
        epsi_ref_ca = ca.SX.sym('epsi_ref')
        vx_ref_ca = ca.SX.sym('vx_ref')
        vy_ref_ca = ca.SX.sym('vy_ref')
        omega_ref_ca = ca.SX.sym('omega_ref')
        delta_ref_ca = ca.SX.sym('delta_ref')
        kappa_ref_ca = ca.SX.sym('kappa_ref')
        oppo_ey_ref_ca = ca.SX.sym('oppo_ey_ref')
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
            oppo_ey_ref_ca,
            accel_ref_ca,
            ddelta_ref_ca
        )

        return ca.SX.sym('P', STATE_DIM+INPUT_DIM+1+1, N+1), reference

    def constructPmatrix(self, state, ref_traj, oppo_states):
        """ Initialize params (reference trajectory states/inputs, curvature, *nearest opponent ey*) """
        track = self.scene_config["track"]
        curvature = track.getCurvature(ref_traj[0,:])
        min_sdist = self.control_config["adversary_dist"]
        oppo_ey = ref_traj[1,:] # If no adversary close, this becomes another ey tracking error term (becomes equivalent to vanilla MPC)
        for agent_ID in oppo_states:
            opp_state = oppo_states[agent_ID]
            opp_s, s = opp_state[0], state[0]
            ds = track.signedSDist(s, opp_s)
            if ds > 0 and np.abs(ds) < min_sdist:
                oppo_ey = opp_state[1] * np.ones((1,ref_traj.shape[1]))
                min_sdist = ds

        state_ref = np.hstack((state.reshape((STATE_DIM,1)), ref_traj[:STATE_DIM,1:]))
        P_mat = np.vstack((state_ref, curvature, oppo_ey, ref_traj[STATE_DIM:]))
        return P_mat

    def stageCostFn(self, st, con, ref, opp=None):  
        """ Define stage cost for modularity (adds cost term to block nearest opponent behind) """
        # Construct state cost matrix
        k_s = self.control_config["opt_k_s"]
        k_ey = self.control_config["adv_opt_k_ey"]
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

        k_ddelta = self.control_config["opt_k_ddelta"]

        # Construct adversarial behavior cost matrix (work to block cars behind)
        k_ey_diff = self.control_config["k_ey_diff"]

        return (st - ref[:STATE_DIM]).T @ Q @ (st - ref[:STATE_DIM]) + \
               (con - ref[-INPUT_DIM:]).T @ R @ (con - ref[-INPUT_DIM:]) +  \
               con[-1].T @ k_ddelta @ con[-1] + \
               (st[1] - ref[STATE_DIM+1]).T @ k_ey_diff @ (st[1] - ref[STATE_DIM+1])

    def terminalCostFn(self, final_st, ref, opp=None): 
        """ Define terminal cost for modularity (adds cost term to block nearest opponent behind)""" 
        # Construct state cost matrix
        k_s = self.control_config["opt_k_s"]
        k_ey = self.control_config["opt_k_ey"]
        k_epsi = self.control_config["opt_k_epsi"]
        k_vx = self.control_config["opt_k_vx"]
        k_vy = self.control_config["opt_k_vy"]
        k_omega = self.control_config["opt_k_omega"]
        k_delta = self.control_config["opt_k_delta"]
        Q = ca.diagcat(k_s, k_ey, k_epsi, k_vx, k_vy, k_omega, k_delta)

        # Construct adversarial behavior cost matrix (work to block cars behind)
        k_ey_diff = self.control_config["k_ey_diff"]

        return (final_st - ref[:STATE_DIM]).T @ Q @ (final_st - ref[:STATE_DIM]) + \
               (final_st[1] - ref[STATE_DIM+1]).T @ k_ey_diff @ (final_st[1] - ref[STATE_DIM+1])


class SafeMPCController(MPCController):
    def __init__(self,  veh_config, scene_config, control_config):
        super().__init__(veh_config, scene_config, control_config)
        self.GP_config = control_config["GP_config"]
        self.gpr = GPRegression(self.GP_config, self.scene_config)
        self.gpr.importGP("gp_models/new/model_5k_250_1-0_ADV.pkl")


    def initPmatrix(self):
        """
        Matrix containing initial state, reference states/inputs, curvature, and nearest max_num_opponents opponent (s,ey) over all timesteps [11 x N+1]
        If less than max_number_opponents, we will assume we set the position of other entries to infinity distance, to not affect the optimization
        First column is initial state/curvature/opponent ey with zeros for inputs, rest of columns are reference state+curvature+input
        Column = [s, ey, epsi, vx, vy, omega, delta, kappa, opp1 s, opp1 ey, opp2 s, opp2 ey, ... , accel, ddelta]
        """
        max_num_opponents = self.control_config["safe_opt_max_num_opponents"]
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

        # Reference symbolic variables
        s_ref_ca = ca.SX.sym('s_ref')
        ey_ref_ca = ca.SX.sym('ey_ref')
        epsi_ref_ca = ca.SX.sym('epsi_ref')
        vx_ref_ca = ca.SX.sym('vx_ref')
        vy_ref_ca = ca.SX.sym('vy_ref')
        omega_ref_ca = ca.SX.sym('omega_ref')
        delta_ref_ca = ca.SX.sym('delta_ref')
        kappa_ref_ca = ca.SX.sym('kappa_ref')
        oppo_pos_ca = ca.SX.sym('oppo_pos', max_num_opponents*2, 1)
        oppo_ey_ref_ca = ca.SX.sym('oppo_ey_ref')
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
            oppo_pos_ca,
            accel_ref_ca,
            ddelta_ref_ca
        )

        return ca.SX.sym('P', STATE_DIM+INPUT_DIM+1+(2*max_num_opponents), N+1), reference

    def constructPmatrix(self, state, ref_traj, oppo_states):
        """ Initialize params (reference trajectory states/inputs, curvature, *nearest opponent states*) """
        track = self.scene_config["track"]
        curvature = track.getCurvature(ref_traj[0,:])

        # Find positions of all agents close enough to consider for safety (up to max_num_opponents agents)
        max_safe_opp_dist = self.control_config["safe_opt_max_opp_dist"]
        max_num_opponents = self.control_config["safe_opt_max_num_opponents"]
        oppo_pos_mat = max_safe_opp_dist*100 * np.ones((max_num_opponents*2, ref_traj.shape[1])) # Initialize (s,ey) to very far away by default to not affect opt
        opp_pos = np.zeros((2, len(oppo_states)))
        opp_future_pos = np.zeros((2, len(oppo_states)))
        # Iterate through to unpack oppo_states into a numpy array
        i = 0
        for agent_ID in oppo_states:
            opp_state = oppo_states[agent_ID]
            future_opp_state = self.inferIntentGP(state, opp_state)
            opp_pos[:,i] = opp_state[:2]
            opp_future_pos[:,i] = future_opp_state[:2]
            i += 1
        if len(oppo_states) <= max_num_opponents: # Less opponents than max, so add them all if close enough
            smallInd = np.arange(0, len(oppo_states), 1)
        else: # Need to look through and find the closest max_num_opponents agents that fulfill criteria to be considered
            smallInd = np.argpartition(opp_future_pos[0,:] - state[0], max_num_opponents)[:max_num_opponents]
        # Loop through and add all positions if close enough
        counter = 0
        for ind in smallInd:
            opp_position = opp_pos[:, ind]
            future_opp_position = opp_future_pos[:, ind]
            curr_opp_s, future_opp_s, s = opp_position[0], future_opp_position[0], state[0]
            ds_curr = track.signedSDist(s, curr_opp_s)
            ds_future = track.signedSDist(s, future_opp_s)
            s_window_lb = -max_safe_opp_dist
            s_window_ub = max_safe_opp_dist
            if np.abs(ds_curr)<max_safe_opp_dist or np.abs(ds_future)<max_safe_opp_dist or (np.sign(ds_curr) != np.sign(ds_future)):
                # Accounts for (1) current opp state unsafe, (2) future opp state unsafe, (3) curr/future state safe, BUT crosses ds=0 in between)
                # Add trajectory to be considered, interpolating between current and future opponent (s,ey)
                oppo_pos_mat[2*counter:2*(counter+1),:] = np.linspace(opp_state[:2], future_opp_state[:2], ref_traj.shape[1]).T
                counter += 1
        state_ref = np.hstack((state.reshape((STATE_DIM,1)), ref_traj[:STATE_DIM,1:]))
        P_mat = np.vstack((state_ref, curvature, oppo_pos_mat, ref_traj[STATE_DIM:]))
        return P_mat

    def configureConstraints(self):
        """ Configure constraints upper and lower bounds (assumes consistent with updateStageCustomConstraints) """
        T = self.control_config["T"]            # Prediction horizon
        freq = self.control_config["opt_freq"]  # Optimization Frequency
        N = int(T*freq)                         # Number of discretization steps

        max_num_opponents = self.control_config["safe_opt_max_num_opponents"]
        # N+1 dynamics constraints, N force constraints, N+1 steps to stay away from max_num_opponents opponents (consider inf norm of s and ey)
        lbg = ca.DM.zeros((STATE_DIM*(N+1) + (N) + (max_num_opponents)*(N+1), 1)) 
        ubg = ca.DM.zeros((STATE_DIM*(N+1) + (N) + (max_num_opponents)*(N+1), 1))
        ubg[STATE_DIM*(N+1):STATE_DIM*(N+1)+(N)] = self.veh_config["downforce_coeff"] * self.veh_config["m"] * 9.81 # Max force constraint
        lbg[STATE_DIM*(N+1)+(N):] = self.control_config["safe_opt_buffer"] + np.max([self.veh_config["lf"]+self.veh_config["lr"], 2*self.veh_config["half_width"]]) # Minimum distance from other agents
        ubg[STATE_DIM*(N+1)+(N):] = ca.inf

        print(lbg)

        return lbg, ubg

    def updateStageCustomConstraints(self, g_custom, st, con, ref):
        """ 
        Add constraints to stay away from nearby agents
        For reference: ref = [s, ey, epsi, vx, vy, omega, delta, kappa, opp1 s, opp1 ey, opp2 s, opp2 ey, ... , accel, ddelta]
        """
        s, ey = st[0], st[1]
        opp_states = ref[STATE_DIM+1:-2] # Should contain max_num_opponents pairs of (s,ey)
        for i in range(self.control_config["safe_opt_max_num_opponents"]):
            opp_s, opp_ey = opp_states[2*i], opp_states[2*i+1]
            g_custom = ca.vertcat(g_custom, ca.norm_inf(ca.vertcat(s - opp_s, ey - opp_ey)))
        return g_custom

    def updateTerminalCustomConstraints(self, g_custom, final_st, ref):
        """ Enforce staying away from nearby agents for terminal state as well """
        return self.updateStageCustomConstraints(g_custom, final_st, None, ref)


    def inferIntentGP(self, state, opp_state):
        vx = opp_state[3]
        return opp_state + np.array([vx*self.control_config["T"]+0.5*0*9,0,0,0,0,0,0]) # TODO: Placeholder
        # gp_predicts = self.gpr.predict(state, opp_state)
        # ds, dey = gp_predicts[:2] # where ds and dey are both from (state - future_opp_state)
        # future_opp_state = np.zeros(opp_state.shape)
        # future_opp_state[:2] = state[:2] - gp_predicts[:2]
        # future_opp_state[2:] = opp_state[2:]
        # return future_opp_state

if __name__ == "__main__":
    from config import *
    veh_config = get_vehicle_config()
    scene_config = get_scene_config(track_type=OVAL_TRACK)
    cont_config = get_controller_config(veh_config, scene_config)
    controller = MPCController(veh_config, scene_config, cont_config)
    controller.getRefTrajectory(3.5, np.linspace(0,0,20))
    controller.computeControl(np.array([300,0,0,0,0,0,0]), [], 0)
