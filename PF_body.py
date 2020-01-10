# -*- coding: utf-8 -*-

#To implement the filter we need to create the particles and the landmarks. 
#We then execute a loop, successively calling predict, update, resampling, 
#and then computing the new state estimate with estimate.

#code creates a uniform and Gaussian distribution
import scipy
import numpy as np
from numpy.random import uniform,randn,random
from numpy.linalg import norm
from math import tan, sin, cos, sqrt,atan2
from filterpy.monte_carlo import systematic_resample 

import matplotlib.pyplot as plt
import scipy.stats


def turn(v, t0, t1, steps):
  return [[v, a] for a in np.linspace(
                 np.radians(t0), np.radians(t1), steps)]  

def move(x, dt, u, wheelbase):
    hdg = x[2]
    vel = u[0]
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) > 0.001: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle) # radius

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        return x + np.array([-r*sinh + r*sinhb, 
                              r*cosh - r*coshb, beta])
    else: # moving in straight line
        return x + np.array([dist*cos(hdg), dist*sin(hdg), 0])


'''If you are passively tracking something (no control input), 
then you would need to include velocity in the state and use that estimate to make the prediction. '''
def create_uniform_particles(x_range, y_range, hdg_range, N):
    '''store N particles in a (N, 3) shaped array. 
    The three columns contain x, y, and heading, in that order.'''
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N) #heading
    particles[:, 2] %= 2 * np.pi
    return particles
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

#Predict Step
def predict(particles, u, std, dt=1.):
    """ move according to control input u (velocity,heading) heading change, velocity)    
    with noise Q (std heading change, std velocity)`"""
    #0.1 meters while turning by 0.007 radians.
    N = len(particles)
    # update heading
    particles[:, 2] += u[1] + (randn(N) * std[0]) 
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[0] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    
#Update Step
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    

#Particle Resampling¶
def simple_resample(particles, weights):
    '''multinomial'''
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)
    
    
def resample_from_index(particles, weights, indexes):
    '''FilterPy doesn't know how your particle filter is implemented, 
    so it cannot generate the new samples. Instead, the algorithms create 
    a numpy.array containing the indexes of the particles that are chosen. 
    Your code needs to perform the resampling step. For example, I used this for the robot:'''
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))    
    
def neff(weights): 
    '''effective N, which approximately measures the number of particles 
    which meaningfully contribute to the probability distribution. '''
    return 1. / np.sum(np.square(weights))

def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


#Computing the State Estimate
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    mean = np.average(particles, weights=weights, axis=0)
    #mean_head = np.average(particles[:,2],weights=weights, axis=0)
    var  = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var

###########body part## assemble all in one
def run_pf1(cmds, N,dt,wheelbase, landmarks, step=10, iters=18, sensor_std_err=.1,sigma_range = 0.3, do_plot=True, 
            plot_estimate=False,xlim=(-60, 60), ylim=(-60, 60),initial_x=None):
    ''' the robot has sensors that measures distance to visible landmarks. '''
    NL = len(landmarks)
    
    plt.figure()
    
    # create particles and weights
    if initial_x is not None: # which means initial position is not given to PF.
        init_flag = 1
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi/4), N=N)
    else:
        init_flag = 0
        particles = create_uniform_particles((-60,60), (-60,60), (0, 6.28), N) #x_range, y_range, hdg_rang
    weights = np.ones(N) / N
    
    if plot_estimate:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)           
        plt.scatter(particles[:, 0], particles[:, 1], 
                    alpha=alpha, color='g')
    
    # plot landmarks
    if len(landmarks) > 0:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], 
                    marker='s', s=60)
    
    if initial_x is not None:
        sim_pos =initial_x
        print(np.shape(sim_pos))
    else:
        sim_pos,_ = estimate(particles,weights)
        sim_pos = np.array([0, 1, 3.14/8])#x_range, y_range, hdg_rang
        print(sim_pos)
    
    track = []
    xs = []
    for i, u in enumerate(cmds):     
        sim_pos = move(sim_pos, dt/step, u, wheelbase) # move for each time film
        track.append(sim_pos)
        if i % step == 0:
            predict(particles, u=u, std=(.2, .05),dt=dt*step)

            x, y = sim_pos[0], sim_pos[1]
            z = []
 
            for lmark in landmarks:
                dx, dy = lmark[0] - x, lmark[1] - y
                d = sqrt(dx**2 + dy**2) + randn()*sigma_range
                z.extend([d])
            update(particles, weights, z=z, R=sensor_std_err,landmarks=landmarks)
            
            # resample if too few effective particles
            if neff(weights) < N/2:
                indexes = systematic_resample(weights)
                resample_from_index(particles, weights, indexes)
                assert np.allclose(weights, 1/N)
            mu, var = estimate(particles, weights)
            xs.append(mu[0:2])
            '''if plot_particles:
                plt.scatter(particles[:, 0], particles[:, 1], 
                            color='k', marker=',', s=1)'''
            #p1 = plt.plot(track[:, 0], track[:,1], color='k', lw=2)
            #p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    track = np.array(track)
    if do_plot:
        p1 = plt.plot(track[:, 0], track[:,1], color='k', lw=3)
    xs = np.array(xs)
    if plot_estimate:
        p2 = plt.plot(xs[:, 0], xs[:,1], color='r', lw=2)
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1
               )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axis('equal')
    plt.title("PF Robot localization")
    print('final position error, variance:\n\t', mu[0:2] - np.array([iters, iters]), var)
    
    plt.savefig('PF_N{0}_SR{1:.2f}_NL{2}_IP{3}.jpg'.format(N,sigma_range,NL,init_flag),dpi = 300)
    plt.show()


