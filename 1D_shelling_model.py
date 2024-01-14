# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:43:47 2024

@author: SENTA
"""

import numpy as np
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:23:02 2023

@author: SENTA
"""

"""I still have to make some small changes like taking into acount that the
   location of the random agent chosen is empty when calculating the
   magnetitzation of the possible vancies  or optimizing the program. Also 
   there must be a better way to show the grid for a larger number of agents"""


import numpy as np
import matplotlib.pyplot as plt
import time 


#---------------------------------------FUNCTIONS------------------------------

def create_grid(N, AGENT_1, VACANTS):
    '''creation of the grid with a random distribution of the agents. Firts it
       creates N+ 1 type agents followed by N- -1 type agent leaving N0 
       vacancies, onces they are all created, it shuffles the array'''
    grid = np.zeros(N, dtype=(int))

    for i in range(N-VACANTS):
        if i < AGENT_1:
            grid[i] = 1
        else:
            grid[i] = -1

    np.random.shuffle(grid)
    
    return grid

def unhappy_agents(N, grid):
    '''Creation of the necessary arrays to run the simulation and
       deciding with agents are unhappy with a tolerance of T=1/2.
       It calculates the magnetitzation of the neighbors of the agent chosen 
       and if it's grater than zero (lower than zero) for the -1 agents (1 
       agents) that means that his neighbors are both 1 agets (-1 agents) and 
       this agent is unhappy.'''
       
    unhappy = np.array([], dtype=(int))
    possible_vacants_0 = np.array([], dtype=(int))
    
    for i in range(N):
        if i == 0: #taking into acount the boundary conditions
            neighbor_mag = grid[N-1] + grid[i+1]
        elif i == N-1:
            neighbor_mag = grid[0] + grid[i-1]
        else:
            neighbor_mag = grid[i+1] + grid[i-1]
            
        if grid[i] == 1:
            
            if neighbor_mag < 0:
                unhappy = np.append(unhappy, i)
        
        elif grid[i] == -1:
            
            if neighbor_mag > 0:
                unhappy = np.append(unhappy, i)
        
        else:
            possible_vacants_0 = np.append(possible_vacants_0, i)
    return unhappy, possible_vacants_0

def cleaning_vacancies(N, possible_vacants_0, grid, random_agent):
    '''Clean the vacancies array with only the ones that will be happier or equal 
       happy for the random chosen agent. It calculates the magnetitzation of 
       the neighbors of the vacancy chosen and if it's grater than zero (lower
       than zero) for the -1 agents (1 agents) that means that his neighbors 
       are both 1 agets (-1 agents) and this vacancy is not possible.'''
       
    k=0
    possible_vacants = np.copy(possible_vacants_0)
    grid_without_agent = np.copy(grid)
    grid_without_agent[random_agent] = 0

    for j in range(len(possible_vacants_0)):  
        if possible_vacants_0[j] == 0: # taking into acount boundary conditions
            neighbor_mag = grid_without_agent[N-1] + grid_without_agent[possible_vacants_0[j]+1]
        elif possible_vacants_0[j] == N-1:
            neighbor_mag = grid_without_agent[0] + grid_without_agent[possible_vacants_0[j]-1]
        else:
            neighbor_mag = grid_without_agent[possible_vacants_0[j]+1] + grid_without_agent[possible_vacants_0[j]-1]
            
        if grid[random_agent] == 1:
            if neighbor_mag < 0:
                k += 1
                possible_vacants = np.delete(possible_vacants, j-k+1)
        
        else:
            if neighbor_mag > 0:
                k += 1
                possible_vacants = np.delete(possible_vacants, j-k+1)
    
    return possible_vacants

def grid_plot(grid_inicial, grid):
    """ It plots the grid in a set of colored rectangles. Red ones if it's a 1
        agent, green ones if it's a -1 agent and white for vacancies"""
    
    fig, ax = plt.subplots(1,2)
    
    for i, val in enumerate(grid_inicial):
        color = 'r' if val == 1 else 'g' if val == -1 else 'w'  
        ax[0].add_patch(plt.Rectangle((i, 0), 1, 1, color=color, alpha=0.6))
    
    ax[0].set_xlim(0, len(grid))
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # ax[0].tick_params(axis='y', which='both', left=False, right=False)
    # ax[0].set_title('initial grid')
    
    for i, val in enumerate(grid):
        color = 'r' if val == 1 else 'g' if val == -1 else 'w'  
        ax[1].add_patch(plt.Rectangle((i, 0), 1, 1, color=color, alpha=0.6))
    
    ax[1].set_xlim(0, len(grid))
    ax[1].set_ylim(0, 1)
    ax[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # ax[1].set_title('frozen state grid')
    
    plt.tight_layout()
    plt.show()

def clusters(grid):
    '''Runs through the grid and count how many clusters there are and their
       size. It returns an array with this information. It looks if there is a 
       agent in each position and the type, if there are from one to any number
       of the same agent concatenated it sum all of them. If it founds a change
       of agent type or a vacncy, it writes the last value to an array, if it 
       is not 0 and starts again'''
       
    cluster = np.array([])
    cluster_agent_1 = np.array([])
    cluster_agent_2 = np.array([])
    cluster_size = 0
    # grid = grid[grid != 0]
    for i in range (len(grid)):
        if grid[i] == 0: 
            if cluster_size != 0:
                cluster = np.append(cluster, cluster_size)
                if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
                if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)
                
            cluster_size = 0
        else:
            if i == 0:
                cluster_size = 1
            else:
                if grid[i-1] == grid[i]:
                    cluster_size += 1 
                elif grid[i-1] != grid[i] and cluster_size != 0:
                    cluster = np.append(cluster, cluster_size)
                    if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
                    if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)
                    cluster_size = 1
                else:
                    cluster_size = 1
    
    if cluster_size != 0:
        if grid[0] == grid[len(grid)-1] and grid[0] != 0:
            cluster[0] += cluster_size
            if grid[i-1] == 1: cluster_agent_1[0] += cluster_size
            if grid[i-1] == -1: cluster_agent_2[0] += cluster_size
        else:
            cluster = np.append(cluster, cluster_size)
            if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
            if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)

    return cluster, cluster_agent_1, cluster_agent_2 

def clusters_2(grid):
    '''Runs through the grid and count how many clusters there are and their
       size. It returns an array with this information. It looks if there is a 
       agent in each position and the type, if there are from one to any number
       of the same agent concatenated it sum all of them. If it founds a change
       of agent type or a vacncy, it writes the last value to an array, if it 
       is not 0 and starts again'''
       
    cluster = np.array([])
    cluster_agent_1 = np.array([])
    cluster_agent_2 = np.array([])
    cluster_size = 0
    grid = grid[grid != 0]
    for i in range (len(grid)):
        if grid[i] == 0: 
            if cluster_size != 0:
                cluster = np.append(cluster, cluster_size)
                if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
                if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)
                
            cluster_size = 0
        else:
            if i == 0:
                cluster_size = 1
            else:
                if grid[i-1] == grid[i]:
                    cluster_size += 1 
                elif grid[i-1] != grid[i] and cluster_size != 0:
                    cluster = np.append(cluster, cluster_size)
                    if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
                    if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)
                    cluster_size = 1
                else:
                    cluster_size = 1
    
    if cluster_size != 0:
        if grid[0] == grid[len(grid)-1] and grid[0] != 0:
            cluster[0] += cluster_size
            if grid[i-1] == 1: cluster_agent_1[0] += cluster_size
            if grid[i-1] == -1: cluster_agent_2[0] += cluster_size
        else:
            cluster = np.append(cluster, cluster_size)
            if grid[i-1] == 1: cluster_agent_1 = np.append(cluster_agent_1, cluster_size)
            if grid[i-1] == -1: cluster_agent_2 = np.append(cluster_agent_2, cluster_size)

    return cluster, cluster_agent_1, cluster_agent_2 

def main_program(density, N, m):
    '''It is the main program. It generates the grid of size N and with a exact
       vacancies density. It counts how many agents are unhappi and tries to 
       move them to a better place until every agent is happy or there is no 
       place to put any unhappy agent.'''

    VACANTS = round(density * N)
    AGENT_1 = round((1 - density + m) * N/2)
    AGENT_2 = N - VACANTS - AGENT_1
    
    # creation of the random grid and plot of the initialized state
    grid = create_grid(N, AGENT_1, VACANTS)
    grid_inicial = np.copy(grid)
    
    # grid_plot(grid)
    for i in range(1000): # for now I am using a for bucle, I am looking a way to put a while insted
                         # if a range reasonably big is chosen there should be no problem
        unhappy, possible_vacants_0 = unhappy_agents(N, grid) # I do this before the if to se if this array is empty
        unhappy_1 = np.copy(unhappy) # keep an untouched copy of the original to calculate the unhappines of the final solution
        
        if len(unhappy) == 0: # is not possible to choose a random element of a empty array
            # print(f'Not unhappy agents found at iteration {i}')
            break
    
        
        else:
            while len(unhappy) != 0:
                random_agent = np.random.choice(unhappy)
                possible_vacants = cleaning_vacancies(N, possible_vacants_0, grid, random_agent) 
                                
                if len(possible_vacants) == 0: #same here with vacancies
                    unhappy= np.delete(unhappy, np.where(unhappy == random_agent))
                                  
                else:
                    random_vacant = np.random.choice(possible_vacants)
                    grid[random_vacant] = grid[random_agent]
                    grid[random_agent] = 0
                    break
            
            if len(possible_vacants) == 0: #if it has tried all unhappy agents and any of them can be moved to a better place we have arrived to a frozen solution
                # print(f'Not possible locations to set any unhappy agent found at iteration {i}')
                break
    
    # grid_plot(grid)
    happines = 1 - len(unhappy_1)/(N-VACANTS)
    cluster, cluster_agent_1, cluster_agent_2 = clusters_2(grid)
    
    return happines, cluster, cluster_agent_1, cluster_agent_2, grid, grid_inicial

def biggest_three(cluster_agent_1, cluster_agent_2):
    
    cluster_1 = np.copy(cluster_agent_1)
    
    if len(cluster_1) != 0:
        arg_first_cluster_1 = np.argmax(cluster_1)
        first_cluster_1 = cluster_1[arg_first_cluster_1]
    
        cluster_1 = np.delete(cluster_1, arg_first_cluster_1)
    else :
        first_cluster_1 = 0
    
    
    if len(cluster_1) != 0:
        arg_second_cluster_1 = np.argmax(cluster_1)
        second_cluster_1 = cluster_1[arg_second_cluster_1]
        
        cluster_1 = np.delete(cluster_1, arg_second_cluster_1)
    else: 
        second_cluster_1 = 0
    
    
    if len(cluster_1) != 0:
            arg_third_cluster_1 = np.argmax(cluster_1)
            third_cluster_1 = cluster_1[arg_third_cluster_1]
    else:
        third_cluster_1 = 0
            
            
    cluster_2 = np.copy(cluster_agent_2)
    
    if len(cluster_2) != 0:
        arg_first_cluster_2 = np.argmax(cluster_2)
        first_cluster_2 = cluster_2[arg_first_cluster_2]
        
        cluster_2 = np.delete(cluster_2, arg_first_cluster_2)
    else :
        first_cluster_2 = 0
        
        
    if len(cluster_2) != 0:
        arg_second_cluster_2 = np.argmax(cluster_2)
        second_cluster_2 = cluster_2[arg_second_cluster_2]
        
        cluster_2 = np.delete(cluster_2, arg_second_cluster_2)
    else: 
        second_cluster_2 = 0
    
    if len(cluster_2) != 0:
            arg_third_cluster_2 = np.argmax(cluster_2)
            third_cluster_2 = cluster_2[arg_third_cluster_2]  
    else:
        third_cluster_2 = 0
            
            
    return first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2

def writing_and_mean(N, m):
    
    N_densities = {0.0:80, 0.1:80, 0.2:79, 0.3:69, 0.4:59, 0.5:49, 0.6:39, 0.7:29, 0.8:19, 0.9:9}
    densities = np.linspace(0.01, N_densities[m]/100, N_densities[m])
    # densities = np.logspace(np.log10(0.01), np.log10(N_densities[m]/100), N_densities[m])
    number_iterations = 100

    start_time = time.perf_counter()
    # for N in range(10, 110, 10):
        
    with open(f'dades_N_{N}_m_{m}.txt', 'w') as file:
        
        for density in densities:
            
            VACANTS = round(density * N)
            AGENT_1 = round((1 - density + m) * N/2)
            AGENT_2 = N - VACANTS - AGENT_1
            

            with open(f'dades_N_{N}_m_{m}_density_{density}.txt', 'w') as file_two:
                mean_happines = 0
                mean_size = 0
                mean_size_1 = 0
                mean_size_2 = 0
                mean_first_cluster_1 = 0
                mean_first_cluster_2 = 0
                mean_second_cluster_1 = 0
                mean_second_cluster_2 = 0
                mean_third_cluster_1 = 0
                mean_third_cluster_2 = 0
                mean_s = 0
                mean_s_1 = 0
                mean_s_2 = 0
                mean_s_quad = 0

                
                for i in range(number_iterations):
                    
                    happines, cluster, cluster_agent_1, cluster_agent_2, grid, grid_inicial = main_program(density, N, m) 
                    
                    first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = biggest_three(cluster_agent_1, cluster_agent_2)
                    
                    s = np.sum(cluster_agent_1**2)/(N-VACANTS)/AGENT_1 + np.sum(cluster_agent_2**2)/(N-VACANTS)/AGENT_2
                    s_quad = s*s
                    s_1 = np.sum(cluster_agent_1**2)/(AGENT_1**2)
                    s_2 = np.sum(cluster_agent_2**2)/(AGENT_2**2)
                    
                    mean_happines += happines
                    mean_size += len(cluster)
                    mean_size_1 += len(cluster_agent_1)
                    mean_size_2 += len(cluster_agent_2)
                    mean_first_cluster_1 += first_cluster_1
                    mean_first_cluster_2 += first_cluster_2
                    mean_second_cluster_1 += second_cluster_1
                    mean_second_cluster_2 += second_cluster_2
                    mean_third_cluster_1 += third_cluster_1
                    mean_third_cluster_2 += third_cluster_2
                    mean_s += s
                    mean_s_1 += s_1
                    mean_s_2 += s_2
                    mean_s_quad += s_quad
                
                    file_two.write(f'{happines}\t {len(cluster)/(N-VACANTS)}\t {len(cluster_agent_1)/AGENT_1}\t {len(cluster_agent_2)/AGENT_2}\t {s}\t {s_1}\t {s_2}\t {first_cluster_1}\t {first_cluster_2}\t {second_cluster_1}\t {second_cluster_2}\t {third_cluster_1}\t {third_cluster_2}\n')
                
                
                
                mean_happines = mean_happines / number_iterations
                mean_size = mean_size / number_iterations
                mean_size_1 = mean_size_1 / number_iterations
                mean_size_2 = mean_size_2 / number_iterations
                mean_first_cluster_1 = mean_first_cluster_1 / number_iterations
                mean_first_cluster_2 = mean_first_cluster_2 / number_iterations
                mean_second_cluster_1 = mean_second_cluster_1 / number_iterations
                mean_second_cluster_2 = mean_second_cluster_2 / number_iterations
                mean_third_cluster_1 = mean_third_cluster_1 / number_iterations
                mean_third_cluster_2 = mean_third_cluster_2 / number_iterations
                mean_s = mean_s / number_iterations
                mean_s_1 = mean_s_1 / number_iterations
                mean_s_2 = mean_s_2 / number_iterations
                mean_s_quad = mean_s_quad / number_iterations
                
                susceptibility = (mean_s_quad - mean_s * mean_s)/density
                
                
            print(round(density, 2), susceptibility, (mean_s_quad - mean_s * mean_s))
            
            file.write(f'{mean_happines}\t {susceptibility}\t {mean_size/(N-VACANTS)}\t {mean_size_1/AGENT_1}\t {mean_size_2/AGENT_2}\t {mean_s}\t {mean_s_1}\t {mean_s_2}\t {mean_first_cluster_1}\t {mean_first_cluster_2}\t {mean_second_cluster_1}\t {mean_second_cluster_2}\t {mean_third_cluster_1}\t {mean_third_cluster_2}\n')
            
    end_time = time.perf_counter()

    print(start_time, end_time, end_time-start_time)
    
#-----------------------------PROGRAM------------------------------------------

N = 100
m = 0.0
density = 0.1
# VACANTS = round(density * N)
# AGENT_1 = round((1 - density + m) * N/2)
# AGENT_2 = N - VACANTS - AGENT_1

# writing_and_mean(N, m)

# s_sin = np.array([])
# s_con = np.array([])
# s_mal = np.array([])

# for i in density:
#     VACANTS = round(i * N)
#     AGENT_1 = round((1 - i + m) * N/2)
#     AGENT_2 = N - VACANTS - AGENT_1
#     # print(f'densiti ={i}, factor de normalizacion = {(N-VACANTS)*AGENT_1}')
happines, cluster, cluster_agent_1, cluster_agent_2, grid, grid_inicial = main_program(density, N, m)
#     cluster_1, cluster_agent_1_1, cluster_agent_2_1  = clusters(grid)
#     cluster_2, cluster_agent_1_2, cluster_agent_2_2  = clusters_2(grid)
grid_plot(grid_inicial, grid)
#     # first_cluster_1, first_cluster_2, second_cluster_1, second_cluster_2, third_cluster_1, third_cluster_2 = biggest_three(cluster_agent_1, cluster_agent_2)
#     s_1 = np.sum(cluster_agent_1_1**2)/(N-VACANTS)/AGENT_1 + np.sum(cluster_agent_2_1**2)/(N-VACANTS)/AGENT_2
#     s_mal_1 = np.sum(cluster_1**2)/(N-VACANTS)/AGENT_1
#     s_2 = np.sum(cluster_agent_1_2**2)/(N-VACANTS)/AGENT_1 + np.sum(cluster_agent_2_2**2)/(N-VACANTS)/AGENT_2
#     s_sin = np.append(s_sin, s_1)
#     s_mal = np.append(s_mal, s_mal_1)
#     s_con = np.append(s_con, s_2)
    
#     # print(f'sin renormalizar: s = {np.sum(cluster_agent_1_1**2)/(N-VACANTS)/AGENT_1 + np.sum(cluster_agent_2_1**2)/(N-VACANTS)/AGENT_2}')
#     # print(f'renormalizado:    s = {np.sum(cluster_agent_1_2**2)/(N-VACANTS)/AGENT_1 + np.sum(cluster_agent_2_2**2)/(N-VACANTS)/AGENT_2}')
#     # print('')


# plt.figure()

# plt.plot(density, s_sin, label ='sin')
# plt.plot(density, s_con, label= 'mal')
# plt.legend()
# plt.grid()
# plt.show()
    
# # happines, cluster, cluster_agent_1, cluster_agent_2, grid, grid_inicial = main_program(density[4], N, m)
# # grid_plot(grid_inicial, grid)
# # cluster_1, cluster_agent_1, cluster_agent_2  = clusters(grid)
# # cluster_2, cluster_agent_1, cluster_agent_2  = clusters_2(grid)
# # print(cluster_1)
# # print(cluster_2)

