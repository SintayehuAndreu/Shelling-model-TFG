# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:12:39 2023

@author: SENTA
"""

import numpy as np
import matplotlib.pyplot as plt





def reading(N, m):
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
    
    VACANTS = np.round(densities * N)
    AGENT_1 = np.round((1 - densities + m) * N/2)
    AGENT_2 = N - VACANTS - AGENT_1
    
    mean_happines = np.array([])
    mean_size = np.array([])
    mean_size_1 = np.array([])
    mean_size_2 = np.array([])
    mean_first_cluster_1 = np.array([])
    mean_first_cluster_2 = np.array([])
    mean_second_cluster_1 = np.array([])
    mean_second_cluster_2 = np.array([])
    mean_third_cluster_1 = np.array([])
    mean_third_cluster_2 = np.array([])
    mean_s = np.array([])
    mean_s_1 = np.array([])
    mean_s_2 = np.array([])
        
      
    with open(f'dades_N_{N}_m_{m}.txt', 'r') as file:
        dades = file.readlines()
    
    for data in dades:
        data_strip = data.strip()
        data_split = data_strip.split('\t ')
        
        happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = data_split
        
        mean_happines = np.append(mean_happines, float(happines))
        mean_size = np.append(mean_size, float(size))
        mean_size_1 = np.append(mean_size_1, float(size_1))
        mean_size_2 = np.append(mean_size_2, float(size_2))
        mean_first_cluster_1 = np.append(mean_first_cluster_1, float(first_cluster_1))
        mean_first_cluster_2 = np.append(mean_first_cluster_2, float(first_cluster_2))
        mean_second_cluster_1 = np.append(mean_second_cluster_1, float(second_cluster_1))
        mean_second_cluster_2 = np.append(mean_second_cluster_2, float(second_cluster_2))
        mean_third_cluster_1 = np.append(mean_third_cluster_1, float(third_cluster_1))
        mean_third_cluster_2 = np.append(mean_third_cluster_2, float(third_cluster_2))
        mean_s = np.append(mean_s, float(s))
        mean_s_1 = np.append(mean_s_1, float(s_1))
        mean_s_2 = np.append(mean_s_2, float(s_2))
        
    return densities, mean_happines, mean_size, mean_size_1, mean_size_2, mean_s, mean_s_1, mean_s_2, mean_first_cluster_1/AGENT_1, mean_first_cluster_2/AGENT_2, mean_second_cluster_1/AGENT_1, mean_second_cluster_2/AGENT_2, mean_third_cluster_1/AGENT_1, mean_third_cluster_2/AGENT_2


def ploting(N, m):
    
    densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading(N, m)
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    
    plt.figure()
    
    plt.plot(size, 1 - happines, label =f'N = {N}')
    plt.title('unhappines')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim(0.01, N_densities[m]/100)
    plt.legend()
    plt.grid()
    
    plt.show()
    
    # fig, axs = plt.subplots(2,2)
    
    # axs[0,0].plot(densities, first_cluster_2, label = f'N = {N}')
    # axs[0,0].set_title('first_cluster_2')
    # # axs[0,0].set_yscale('log')
    # # axs[0,0].set_xscale('log')
    # axs[0,0].set_xlim(0.01,N_densities[m]/100)
    # axs[0,0].legend()
    # axs[0,0].grid()
    
    # axs[1,0].plot(densities, second_cluster_2, label = f'N = {N}')
    # axs[1,0].set_title('second_cluster_2')
    # # axs[1,0].set_yscale('log')
    # # axs[1,0].set_xscale('log')
    # axs[1,0].set_xlim(0.01,N_densities[m]/100)
    # axs[1,0].legend()
    # axs[1,0].grid()
    
    # axs[0,1].plot(densities, third_cluster_2, label = f'N = {N}')
    # axs[0,1].set_title('third_cluster_2')
    # # axs[0,1].set_yscale('log')
    # # axs[0,1].set_xscale('log')
    # axs[0,1].set_xlim(0.01,N_densities[m]/100)
    # axs[0,1].legend()
    # axs[0,1].grid()
    
    # axs[1,1].plot(densities, size, label = f'N = {N}')
    # axs[1,1].set_title('size')
    # # axs[1,1].set_yscale('log')
    # # axs[1,1].set_xscale('log')
    # axs[1,1].set_xlim(0.01,N_densities[m]/100)
    # axs[1,1].legend()
    # axs[1,1].grid()
        
    # plt.tight_layout()
    # plt.show()
    
    # fig, axs = plt.subplots(2,2)
    
    # axs[0,0].plot(densities, first_cluster_1, label = f'N = {N}')
    # axs[0,0].set_title('first_cluster_1')
    # # axs[0,0].set_yscale('log')
    # # axs[0,0].set_xscale('log')
    # axs[0,0].set_xlim(0.01,N_densities[m]/100)
    # axs[0,0].legend()
    # axs[0,0].grid()
    
    # axs[1,0].plot(densities, second_cluster_1, label = f'N = {N}')
    # axs[1,0].set_title('second_cluster_1')
    # # axs[1,0].set_yscale('log')
    # # axs[1,0].set_xscale('log')
    # axs[1,0].set_xlim(0.01,N_densities[m]/100)
    # axs[1,0].legend()
    # axs[1,0].grid()
    
    # axs[0,1].plot(densities, third_cluster_1, label = f'N = {N}')
    # axs[0,1].set_title('third_cluster_1')
    # # axs[0,1].set_yscale('log')
    # # axs[0,1].set_xscale('log')
    # axs[0,1].set_xlim(0.01,N_densities[m]/100)
    # axs[0,1].legend()
    # axs[0,1].grid()
    
    # axs[1,1].plot(densities, s, label = f'N = {N}')
    # axs[1,1].set_title('s')
    # # axs[1,1].set_yscale('log')
    # # axs[1,1].set_xscale('log')
    # axs[1,1].set_xlim(0.01,N_densities[m]/100)
    # axs[1,1].legend()
    # axs[1,1].grid()
        
    # plt.tight_layout()
    # plt.show()
    
    # fig, axs = plt.subplots(2,2)
    
    # axs[0,0].plot(densities, size_1, label = f'N = {N}')
    # axs[0,0].set_title('size_1')
    # # axs[0,0].set_yscale('log')
    # # axs[0,0].set_xscale('log')
    # axs[0,0].set_xlim(0.01,N_densities[m]/100)
    # axs[0,0].legend()
    # axs[0,0].grid()
    
    # axs[1,0].plot(densities, size_2, label = f'N = {N}')
    # axs[1,0].set_title('size_2')
    # # axs[1,0].set_yscale('log')
    # # axs[1,0].set_xscale('log')
    # axs[1,0].set_xlim(0.01,N_densities[m]/100)
    # axs[1,0].legend()
    # axs[1,0].grid()
    
    # axs[0,1].plot(densities, s_1, label = f'N = {N}')
    # axs[0,1].set_title('s_1')
    # # axs[0,1].set_yscale('log')
    # # axs[0,1].set_xscale('log')
    # axs[0,1].set_xlim(0.01,N_densities[m]/100)
    # axs[0,1].legend()
    # axs[0,1].grid()
    
    # axs[1,1].plot(densities, s_2, label = f'N = {N}')
    # axs[1,1].set_title('s_2')
    # # axs[1,1].set_yscale('log')
    # # axs[1,1].set_xscale('log')
    # axs[1,1].set_xlim(0.01,N_densities[m]/100)
    # axs[1,1].legend()
    # axs[1,1].grid()
        
    # plt.tight_layout()
    # plt.show()

N = 3000
m = 0.4

ploting(N, m)