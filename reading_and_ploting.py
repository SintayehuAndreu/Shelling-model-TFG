# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:12:39 2023

@author: SENTA
"""

import numpy as np
import matplotlib.pyplot as plt





def reading_1(N, m):
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
    # densities = np.logspace(np.log10(0.01), np.log10(N_densities[m]/100), N_densities[m])
    
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
        
    return densities, mean_happines, mean_size, mean_size_1, mean_size_2, mean_s*((N-VACANTS)**2)/2/(N-VACANTS)/AGENT_1, mean_s_1/2, mean_s_2/2, mean_first_cluster_1/AGENT_1, mean_first_cluster_2/AGENT_2, mean_second_cluster_1/AGENT_1, mean_second_cluster_2/AGENT_2, mean_third_cluster_1/AGENT_1, mean_third_cluster_2/AGENT_2

def reading_2(N, m):
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
    # densities = np.logspace(np.log10(0.01), np.log10(N_densities[m]/100), N_densities[m])
    
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
    mean_susceptibility = np.array([])
        
      
    with open(f'dades_N_{N}_m_{m}.txt', 'r') as file:
        dades = file.readlines()
    
    for data in dades:
        data_strip = data.strip()
        data_split = data_strip.split('\t ')
        
        happines, susceptibility, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = data_split
        
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
        mean_susceptibility = np.append(mean_susceptibility, float(susceptibility))
        
    return densities, mean_happines, mean_susceptibility, mean_size, mean_size_1, mean_size_2, mean_s, mean_s_1, mean_s_2, mean_first_cluster_1/AGENT_1, mean_first_cluster_2/AGENT_2, mean_second_cluster_1/AGENT_1, mean_second_cluster_2/AGENT_2, mean_third_cluster_1/AGENT_1, mean_third_cluster_2/AGENT_2

def cal_susceptivility(N, m):
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
    # densities = np.logspace(np.log10(0.01), np.log10(N_densities[m]/100), N_densities[m])
    number_iterations = 100
    
    
    
    
    array_s = np.array([])
    array_s_quad = np.array([])
    array_susceptibility = np.zeros(len(densities))
        
    for density in densities:
        mean_s = 0
        mean_s_quad = 0
        
        # VACANTS = np.round(density * N)
        # AGENT_1 = np.round((1 - density + m) * N/2)
        # AGENT_2 = N - VACANTS - AGENT_1   
        
        with open(f'dades_N_{N}_m_{m}_density_{density}.txt', 'r') as file:
            dades = file.readlines()
        
        for data in dades:
            data_strip = data.strip()
            data_split = data_strip.split('\t ')
            
            happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = data_split
            
            mean_s += 1-float(happines)
            mean_s_quad += (1-float(happines))*(1-float(happines))
        
        mean_s = mean_s / number_iterations
        mean_s_quad = mean_s_quad / number_iterations
        
        array_s = np.append(array_s, mean_s)
        array_s_quad = np.append(array_s_quad, mean_s_quad)
    
    for i in range(len(densities)):
        
        array_susceptibility[i] = (array_s_quad[i] - array_s[i]*array_s[i])/densities[i]
    
    return array_susceptibility

def histograma(N,m,density):
    
    VACANTS = np.round(density * N)
    AGENT_1 = np.round((1 - density + m) * N/2)
    AGENT_2 = N - VACANTS - AGENT_1  
    
    happiness = np.array([])
    with open(f'dades_N_{N}_m_{m}_density_{density}.txt', 'r') as file:
        dades = file.readlines()
    
    for data in dades:
        data_strip = data.strip()
        data_split = data_strip.split('\t ')
        
        happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = data_split
        happiness = np.append(happiness, 1-float(happines))
    
    return happiness
        
        

def ploting(N, m):
    
    densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_1(N, m)
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    
    plt.figure()
    
    plt.plot(densities, 1 - happines, label = f'N = {N}')
    plt.title('unhappines')
    plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim(0.01, N_densities[m]/100)
    plt.legend()
    plt.grid()
    
    plt.show()
    
    fig, axs = plt.subplots(2,2)
    
    axs[0,0].plot(densities, first_cluster_2, label = f'N = {N}')
    axs[0,0].set_title('first_cluster_2')
    # axs[0,0].set_yscale('log')
    # axs[0,0].set_xscale('log')
    axs[0,0].set_xlim(0.01,N_densities[m]/100)
    axs[0,0].legend()
    axs[0,0].grid()
    
    axs[1,0].plot(densities, second_cluster_2, label = f'N = {N}')
    axs[1,0].set_title('second_cluster_2')
    # axs[1,0].set_yscale('log')
    # axs[1,0].set_xscale('log')
    axs[1,0].set_xlim(0.01,N_densities[m]/100)
    axs[1,0].legend()
    axs[1,0].grid()
    
    axs[0,1].plot(densities, third_cluster_2, label = f'N = {N}')
    axs[0,1].set_title('third_cluster_2')
    # axs[0,1].set_yscale('log')
    # axs[0,1].set_xscale('log')
    axs[0,1].set_xlim(0.01,N_densities[m]/100)
    axs[0,1].legend()
    axs[0,1].grid()
    
    axs[1,1].plot(densities, size, label = f'N = {N}')
    axs[1,1].set_title('size')
    # axs[1,1].set_yscale('log')
    # axs[1,1].set_xscale('log')
    axs[1,1].set_xlim(0.01,N_densities[m]/100)
    axs[1,1].legend()
    axs[1,1].grid()
        
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(2,2)
    
    axs[0,0].plot(densities, first_cluster_1, label = f'N = {N}')
    axs[0,0].set_title('first_cluster_1')
    # axs[0,0].set_yscale('log')
    # axs[0,0].set_xscale('log')
    axs[0,0].set_xlim(0.01,N_densities[m]/100)
    axs[0,0].legend()
    axs[0,0].grid()
    
    axs[1,0].plot(densities, second_cluster_1, label = f'N = {N}')
    axs[1,0].set_title('second_cluster_1')
    # axs[1,0].set_yscale('log')
    # axs[1,0].set_xscale('log')
    axs[1,0].set_xlim(0.01,N_densities[m]/100)
    axs[1,0].legend()
    axs[1,0].grid()
    
    axs[0,1].plot(densities, third_cluster_1, label = f'N = {N}')
    axs[0,1].set_title('third_cluster_1')
    # axs[0,1].set_yscale('log')
    # axs[0,1].set_xscale('log')
    axs[0,1].set_xlim(0.01,N_densities[m]/100)
    axs[0,1].legend()
    axs[0,1].grid()
    
    axs[1,1].plot(densities, s, label = f'N = {N}')
    axs[1,1].set_title('s')
    # axs[1,1].set_yscale('log')
    # axs[1,1].set_xscale('log')
    axs[1,1].set_xlim(0.01,N_densities[m]/100)
    axs[1,1].legend()
    axs[1,1].grid()
        
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(2,2)
    
    axs[0,0].plot(densities, size_1, label = f'N = {N}')
    axs[0,0].set_title('size_1')
    # axs[0,0].set_yscale('log')
    # axs[0,0].set_xscale('log')
    axs[0,0].set_xlim(0.01,N_densities[m]/100)
    axs[0,0].legend()
    axs[0,0].grid()
    
    axs[1,0].plot(densities, size_2, label = f'N = {N}')
    axs[1,0].set_title('size_2')
    # axs[1,0].set_yscale('log')
    # axs[1,0].set_xscale('log')
    axs[1,0].set_xlim(0.01,N_densities[m]/100)
    axs[1,0].legend()
    axs[1,0].grid()
    
    axs[0,1].plot(densities, s_1, label = f'N = {N}')
    axs[0,1].set_title('s_1')
    # axs[0,1].set_yscale('log')
    # axs[0,1].set_xscale('log')
    axs[0,1].set_xlim(0.01,N_densities[m]/100)
    axs[0,1].legend()
    axs[0,1].grid()
    
    axs[1,1].plot(densities, s_2, label = f'N = {N}')
    axs[1,1].set_title('s_2')
    # axs[1,1].set_yscale('log')
    # axs[1,1].set_xscale('log')
    axs[1,1].set_xlim(0.01,N_densities[m]/100)
    axs[1,1].legend()
    axs[1,1].grid()
        
    plt.tight_layout()
    plt.show()

def unhappiness_teorica(density):
    density_1 = (1 - density + m)/2
    density_2 = (1 - density - m)/2
    
    density_U_1 = density_1*(density_2)**2
    density_U_2 = density_2*(density_1)**2
    
    density_F_2 = density_1*(2*density*density_2 + density_2**2)
    
    density_V_1 = density*(2*density*density_1 + density_1**2)
    density_V_2 = density*(2*density*density_2 + density_2**2)
    
    density_f_2 = density*(1 - density_1**2 - 2*density*density_1)
    
    density_VVV = density**3
    
    return density_U_2 - density_U_1 - density_f_2

def unhappiness_teorica_2(density):
    density_1 = (1 - density + m)/2
    density_2 = (1 - density - m)/2
    
    density_U_1 = density_1*(density_2)**2
    density_U_2 = density_2*(density_1)**2
    
    density_F_2 = density_1*(2*density*density_2 + density_2**2)
    
    density_V_1 = density*(2*density*density_1 + density_1**2)
    density_V_2 = density*(2*density*density_2 + density_2**2)
    
    density_f_2 = density*(1 - density_1**2)
    
    density_VVV = density**3
    
    return density_U_2 - density_U_1 - density_f_2
    
N = 3000
m = 0.4
N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
densities_2 = np.linspace(0.0001, 1, 10000)

n = (100, 500, 1000, 2000, 3000)
M = (0.0, 0.2, 0.4)
# ploting(N, m)



plt.figure()

for m in M:
    susceptivility_unhappines = cal_susceptivility(N, m)
    # densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_1(N, m)
    densities, happines, susceptivility, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_2(N, m)
# unhappines = histograma(N,m,0.05999999999999999)
    t_unhappines = unhappiness_teorica(densities)
    # t_unhappines_2 = unhappiness_teorica_2(densities_2)

    # plt.plot(densities, s, label = f'N = {N}')
    plt.errorbar(densities, 1- happines, susceptivility_unhappines*densities, 0, label = f'm = {m}')
    plt.plot(densities, t_unhappines,'--',  label = f'theoretical m = {m}')
# plt.plot(densities_2, t_unhappines_2, label = f'teoric original')
    

plt.title(f'unhappinss for N={N}')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.00001, 0.1)
plt.xlim(0.01, 0.85)
plt.xlabel('$\\rho_0$')
plt.ylabel('$\\langle u \\rangle$')


plt.legend()
plt.grid()
plt.tight_layout()
    
plt.show()

N = 3000
m = 0.4

plt.figure()

for m in M:
    susceptivility_unhappines = cal_susceptivility(N, m)
    # densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_1(N, m)
    densities, happines, susceptivility, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_2(N, m)
# unhappines = histograma(N,m,0.05999999999999999)
    t_unhappines = unhappiness_teorica(densities)
    # t_unhappines_2 = unhappiness_teorica_2(densities_2)

    # plt.plot(densities, s, label = f'N = {N}')
    plt.errorbar(densities, s, susceptivility*densities, 0, label = f'm = {m}')
# plt.plot(densities, t_unhappines,'--',  label = f'theoretical')
# plt.plot(densities_2, t_unhappines_2, label = f'teoric original')
    

plt.title(f'segregation for N={N}')
plt.xscale('log')
plt.yscale('log')
# plt.ylim(-0.001, 0.08)
plt.xlim(0.01, 0.85)
plt.xlabel('$\\rho_0$')
plt.ylabel('$\\langle s \\rangle$')


plt.legend()
plt.grid()
plt.tight_layout()
    
plt.show()

N = 3000
m = 0.4

plt.figure()

for N in n:
    susceptivility_unhappines = cal_susceptivility(N, m)
    # densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_1(N, m)
    densities, happines, susceptivility, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_2(N, m)
# unhappines = histograma(N,m,0.05999999999999999)
    t_unhappines = unhappiness_teorica(densities)
    # t_unhappines_2 = unhappiness_teorica_2(densities_2)

    # plt.plot(densities, s, label = f'N = {N}')
    plt.errorbar(densities, 1- happines, susceptivility_unhappines*densities, 0, label = f'N = {N}')
plt.plot(densities, t_unhappines,'--',  label = f'theoretical')
# plt.plot(densities_2, t_unhappines_2, label = f'teoric original')
    

plt.title(f'unhappinss for m={m}')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.00001, 0.1)
plt.xlim(0.01, 0.65)
plt.xlabel('$\\rho_0$')
plt.ylabel('$\\langle u \\rangle$')


plt.legend()
plt.grid()
plt.tight_layout()
    
plt.show()

N = 3000
m = 0.4

plt.figure()

for N in n:
    susceptivility_unhappines = cal_susceptivility(N, m)
    # densities, happines, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_1(N, m)
    densities, happines, susceptivility, size, size_1, size_2, s, s_1, s_2, first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = reading_2(N, m)
# unhappines = histograma(N,m,0.05999999999999999)
    t_unhappines = unhappiness_teorica(densities)
    # t_unhappines_2 = unhappiness_teorica_2(densities_2)

    # plt.plot(densities, s, label = f'N = {N}')
    plt.errorbar(densities, s, susceptivility*densities, 0, label = f'N = {N}')
# plt.plot(densities, t_unhappines,'--',  label = f'theoretical')
# plt.plot(densities_2, t_unhappines_2, label = f'teoric original')
    

plt.title(f'segregation for m={m}')
plt.xscale('log')
plt.yscale('log')
# plt.ylim(-0.001, 0.08)
plt.xlim(0.01, 0.65)
plt.xlabel('$\\rho_0$')
plt.ylabel('$\\langle s \\rangle$')


plt.legend()
plt.grid()
plt.tight_layout()
    
plt.show()


# plt.figure()

# H, S = np.meshgrid(happines, s)

# plt.imshow(H)
# plt.show

