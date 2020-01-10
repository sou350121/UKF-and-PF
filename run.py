# -*- coding: utf-8 -*-

import numpy as np 
from PF_body import *
from UKF_body import *

# case 1 
def cmds_case1():
    cmds = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
    cmds.extend([cmds[-1]]*700)
    return cmds

# case 2 

# accelerate from a stop
def cmds_case2():
    cmds = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
    cmds.extend([cmds[-1]]*50)
    
    # turn left
    v = cmds[-1][0]
    cmds.extend(turn(v, 0, 2, 15))
    cmds.extend([cmds[-1]]*100)
    
    #turn right
    cmds.extend(turn(v, 2, -2, 15))
    cmds.extend([cmds[-1]]*200)
    
    cmds.extend(turn(v, -2, 0, 15))
    cmds.extend([cmds[-1]]*150)
    
    cmds.extend(turn(v, 0, 1, 25))
    cmds.extend([cmds[-1]]*100)
    return cmds

# case 3
def cmds_case3():
    cmds = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
    cmds.extend([cmds[-1]]*50)
    
    # turn left
    v = cmds[-1][0]
    cmds.extend(turn(v, 0, 3, 15))
    cmds.extend([cmds[-1]]*100)
    
    #turn right
    cmds.extend(turn(v, 2, -3, 15))
    cmds.extend([cmds[-1]]*200)
    
    cmds.extend(turn(v, -2, 0, 15))
    cmds.extend([cmds[-1]]*150)
    
    cmds.extend(turn(v, 0, 5, 25))
    cmds.extend([cmds[-1]]*100)
    return cmds

landmarks = np.array([[5, 5], [15, 5], [25, 15],[15,25] ,[40, 25], [55,10]])
cmds = cmds_case2()

dt = 0.1
wheelbase = 0.5
sigma_range=0.3
sigma_bearing=0.1

from numpy.random import seed
seed(900) 
init_pos = [0, 1, 3.14/8]
N = [100,1000,10000,50000]
#for n in N:
    #run_pf1(cmds=cmds,landmarks=landmarks[:7], N=n, step=1, iters=10,
    #sigma_range=0.3,plot_particles=True,initial_x=init_pos)

#run_pf1(initial_x=init_pos,cmds=cmds,landmarks=landmarks[:],dt=dt, wheelbase=wheelbase, 
   #     N=8000, step=1, iters=3,sigma_range=0.3,do_plot=True,plot_estimate=True)
'''
for sigma_bearing in [0.001,0.01,0.1,0.3,0.6,0.9]:
    ukf = run_ukf(init_pos=init_pos ,cmds=cmds, landmarks=landmarks[:3], dt=dt,wheelbase=wheelbase,
                  sigma_vel=0.1, sigma_steer=np.radians(1),sigma_range=0.3, 
                  sigma_bearing=sigma_bearing, step=1,ellipse_step=20,do_plot=True,plot_estimate=True)
'''
'''
for sigma_range in [0.001,0.01,0.1,0.3,0.6,0.9]:
    ukf = run_ukf(init_pos=init_pos ,cmds=cmds, landmarks=landmarks[:3], dt=dt,wheelbase=wheelbase,
                  sigma_vel=0.1, sigma_steer=np.radians(1),sigma_range=sigma_range, 
                  sigma_bearing=0.1, step=1,ellipse_step=20,do_plot=True,plot_estimate=True)
'''
'''
for NL in range(len(landmarks)):
    ukf = run_ukf(init_pos=init_pos ,cmds=cmds, landmarks=landmarks[:NL+1], dt=dt,wheelbase=wheelbase,
                  sigma_vel=0.1, sigma_steer=np.radians(1),sigma_range=0.35, 
                  sigma_bearing=0.1, step=1,ellipse_step=20,do_plot=True,plot_estimate=True)    
   '''

'''
for sigma_range in [0.001,0.01,0.1,0.3,0.6,0.9]:
    run_pf1(initial_x=init_pos ,cmds=cmds, landmarks=landmarks[:3], dt=dt, wheelbase=wheelbase,
            N=8000, step=1, iters=3,sigma_range=sigma_range, do_plot=True,plot_estimate=True)
       '''

for NL in range(len(landmarks)):
    run_pf1(initial_x=init_pos ,cmds=cmds, landmarks=landmarks[:NL+1], dt=dt, wheelbase=wheelbase,
            N=8000, step=1, iters=3,sigma_range=sigma_range, do_plot=True,plot_estimate=True)
    
'''
for NP in N:
    run_pf1(cmds=cmds,landmarks=landmarks[:3],dt=dt, wheelbase=wheelbase, 
            N=NP, step=1, iters=3,sigma_range=0.35,do_plot=True,plot_estimate=True)
    '''
'''
for st in np.linspace(4,60,8):
    ukf = run_ukf(init_pos=init_pos ,cmds=cmds, landmarks=landmarks[:3], dt=dt,wheelbase=wheelbase,
                  sigma_vel=0.1, sigma_steer=np.radians(1),sigma_range=0.3, 
                  sigma_bearing=0.1, step=1,ellipse_step=10,do_plot=True,plot_estimate=True)    
'''