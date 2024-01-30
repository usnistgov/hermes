# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:38:13 2022

@author: asm6
"""
import numpy as np

def Euler_to_quat_noise(Euler_angles, sigma_psi1, sigma_phi, sigma_psi2):
    #Given Euler angles in Bunge Convention (psi1, phi, psi2),
    # and some Gausian noise on those angles,
    # Calculate the noise in the quaternions:
    
    noise_psi1 = np.random.normal(0, sigma_psi1)
    noise_phi  = np.random.normal(0, sigma_phi)
    noise_psi2 = np.random.normal(0, sigma_psi2)
    
    psi1 = Euler_angles[:,0].reshape(-1,1) + noise_psi1
    phi  = Euler_angles[:,1].reshape(-1,1) + noise_phi
    psi2 = Euler_angles[:,2].reshape(-1,1) + noise_psi2
    
    q0        =  np.cos(0.5*phi)*np.cos(0.5*(psi1+psi2))
    
    q1        = -np.sin(0.5*phi)*np.cos(0.5*(psi1-psi2))       
    
    q2        = -np.sin(0.5*phi)*np.sin(0.5*(psi1-psi2)) 
    
    q3        = -np.cos(0.5*phi)*np.sin(0.5*(psi1+psi2))
    
    P = q0/np.abs(q0) #if q0 is negative, flip the sign of the quaternion

    Noisy_Quaternions = P*np.concatenate((q0,q1,q2,q3), 1)
    
    return Noisy_Quaternions
    