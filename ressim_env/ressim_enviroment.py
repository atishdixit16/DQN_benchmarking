""" Transient two-phase (oil-water) flow Environment for RL studies"""

import numpy as np
import functools
import matplotlib.pyplot as plt
import csv

from gym import spaces

import ressim_env.ressim as ressim
import ressim_env.utils as utils


# parameters

class resSimEnv:

    def __init__(self,
                action_steps,
                patm = 1.0,
                nx = 50,
                ny = 50,
                lx = 2.0,
                ly = 2.0,
                mu_w = 1.0,
                mu_o = 60,
                s_wir = 0.2,
                s_oir = 0.2,
                k = np.ones((50,50)),
                dt = 5e-4,
                n_steps = 60):

        np.random.seed(43)  # for reproducibility

        self.patm = patm
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.mu_w, self.mu_o = mu_w, mu_w  # viscosities
        self.s_wir, self.s_oir = s_wir, s_oir  # irreducible saturations
        self.dt = dt  # timestep
        self.n_steps = n_steps

        self.grid = ressim.Grid(nx=self.nx, ny=self.ny, lx=self.lx, ly=self.ly)  # unit square, 64x64 grid
        self.k = np.ones(self.grid.shape) #uniform permeability

        # initial flow rates
        self.q = np.zeros(self.grid.shape)
        self.Q_SCALE = 1.0
        self.q[ 0, 0]=-0.5 # producer 1 
        self.q[-1, 0]=-0.5 # producer 2
        self.q[ 0,-1]=0.5 # injector 1
        self.q[-1,-1]=0.5 # injector 2
        self.q *= self.Q_SCALE

        self.phi = np.ones(self.grid.shape)*0.1  # uniform porosity
        self.s_init = np.ones(self.grid.shape) * self.s_wir  # initial water saturation equals s_wir
        self.s_load = self.s_init

        # RL parameters
        self.observation_space = spaces.Discrete(8)
        self.action_space = spaces.Discrete(int(action_steps**2)) # should be a perfect square number

        # Model definition

        self.mobi_fn = functools.partial(utils.quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        self.lamb_fn = functools.partial(utils.lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(utils.f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        # (Optional) derivative of water fractional flow
        # This is to compute the jacobian of the residual to accelerate the
        # saturation solver. If not provided, the jacobian is approximated in the
        # solver.
        self.df_fn = functools.partial(utils.df_fn, mobi_fn=self.mobi_fn)
        # instantiate solvers
        # solve pressure

        self.solverP = ressim.PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = ressim.SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)
        
        self.solverP.s = self.s_load
        self.solverP.step()

        # solve saturation
        self.solverS.v = self.solverP.v
        self.solverS.step(self.dt)

        self.p_init = self.solverP.p
        self.p_load = self.p_init

        with open('record_RL_iterations.csv', mode='w') as file: 
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
            writer.writerow(['Episode', 'Run', 'state_i2_s', 'state_i1_s' , 'state_p2_s', 'state_p1_s',  'state_i2_p', 'state_i1_p' , 'state_p2_p', 'state_p1_p', 'q_p1', 'q_p2', 'q_i1', 'q_i2', 'reward', 'cum_reward'])

    
    def step(self, action):
        #solve timesteps

        # action space (source term) is considered discrete [0,0.25,0.5,0.75,1.0] for producer 1 (negative in case of producers) and injector 1 
        # Thus, total action space consists of 25 values ( each combination )

        # source term for producer 1: q[0,0]
        self.q[0,0] = ( -1 / ( self.action_space.n**(1/2) - 1 ) ) * (int(action / self.action_space.n**(1/2) ))
        self.q[-1,0] = -1 - self.q[0,0] # since q[0,0] + q[-1,0] = -1

        # source term for injectior 1: q[0,-1]
        self.q[0,-1] = ( 1 / ( self.action_space.n**(1/2) - 1 ) ) * ( action % self.action_space.n**(1/2) )
        self.q[-1,-1] = 1 - self.q[0,-1] # since q[0,-1] + q[-1,-1] = 1

        self.q *= self.Q_SCALE

        # solve pressure

        self.solverP = ressim.PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = ressim.SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        for _ in range(self.n_steps):
            self.solverP.s = self.s_load
            self.solverP.step()

            # solve saturation
            self.solverS.v = self.solverP.v
            self.solverS.step(self.dt)

        self.s_load = self.solverS.s
        self.p_load = self.solverP.p

        # reward is designed according to problem statement (objective of the optimization)
        # For example,
        # The task, here, is to maximize the total production 
        reward =  -self.q[0,0] * (1 - self.s_load[0,0]) + -self.q[-1,0] * ( 1 - self.s_load[-1,0] )

        # done is always zero since this is a continous task
        done = 0

        # states are represented by values of sturation and pressure at producers and injectors
        state = np.array( [ self.s_load[-1,-1], self.s_load[0,-1],self.s_load[-1,0],self.s_load[0,0], self.p_load[-1,-1], self.p_load[0,-1],self.p_load[-1,0],self.p_load[0,0] ] )
        return state, reward, done, {}

    def reset(self):

        self.q[0,0]=-0.5 # producer 1 
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=0.5 # injector 1
        self.q[-1,-1]=0.5  # injector 2

        self.q *= self.Q_SCALE

        self.s_load = self.s_init

        state = np.array( [ self.s_init[-1,-1], self.s_init[0,-1],self.s_init[-1,0],self.s_init[0,0], self.p_init[-1,-1], self.p_init[0,-1],self.p_init[-1,0],self.p_init[0,0] ] )
        return state

    def render(self, episode, step):
        # plot variable
        if (episode==0 and step==0):
            plt.ion()
            self.fig = plt.figure()
            self.ax1 = self.fig.add_axes([0.1,0.1,0.7,0.8])
            self.ax2 = self.fig.add_axes([0.85, 0.1, 0.05, 0.8])

        plt.sca(self.ax1)
        plt.title("Episode: "+str(episode)+" Step: "+str(step)+"\n"+"Q_p1 = "+str(self.q[0,0])+"; Q_p2 = "+str(self.q[-1,0])+"; Q_i1 = "+str(self.q[-1,-1])+"; Q_i2 = "+str(self.q[0,-1]), loc = 'left')
        k = self.ax1.contourf(self.s_load, levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.colorbar(k, cax = self.ax2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record(self, episode, step, reward, cumReward):
        with open('record_RL_iterations.csv','a') as fd:
            fd.write('\n'+str(episode)+','+str(step)+','+str(self.s_load[-1,-1])+','+str(self.s_load[0,-1])+','+str(self.s_load[-1,0])+','+str(self.s_load[0,0])+','+ str(self.p_load[-1,-1]) + ',' + str(self.p_load[0,-1])+',' + str(self.p_load[-1,0]) +',' + str(self.p_load[0,0]) + ',' +str(self.q[0,0])+','+str(self.q[-1,0])+','+str(self.q[-1,-1])+','+str(self.q[0,-1]) + ','+ str(reward) + ',' + str(cumReward) )