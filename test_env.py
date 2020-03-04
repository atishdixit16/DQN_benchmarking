import numpy as np
from ressim_env.ressim_enviroment import resSimEnv

def gitter(w):
    stencil = [-5,+5,-1,+1, 0]
    w = np.min ( [ np.max ( [ w + stencil[np.random.randint(5)] , 0 ] ) , 24 ] )
    
    return w


# test case for evaluating cumulative reward for each action
env = resSimEnv(5)
cumRewards = []
for i in range(1):
    state = env.reset()
    for j in range(10):
        # env.render(i,j)
        # print(env.q[0,0], env.q[-1,0], env.q[0,-1], env.q[-1,-1])
        cumR = 0.0
        action = j #np.random.randint(low=0, high=25)
        # print (action)
        for count in range(100):
            # env.render(j,count)
            state, reward, done, info = env.step(action)
            cumR += reward
        cumRewards.append(cumR)
        print(action, cumR)

xx


# test case for random action change
env = resSimEnv()
cumR10 = []
# action = np.random.randint(25)
for i in range(100):
    cumR = 0.0
    state = env.reset()
    for j in range(100):
        env.render(i,j)
        # print(env.q[0,0], env.q[-1,0], env.q[0,-1], env.q[-1,-1])
        action = np.random.randint(25)
        # print (action)
        state, reward, done, info = env.step(action)
        cumR += reward
    cumR10.append(cumR)
    print(i,cumR)
        # env.render(i,j)

# run this part saperately 

import numpy as np
import matplotlib.pyplot as plt 
cumR4 = np.load('cum4.npy')
cumR5 = np.load('cum5.npy')
cumR10 = np.load('cum10.npy')
cumR20 = np.load('cum20.npy')
cumRall = np.load('cumAll.npy')

data = np.column_stack((cumR4, cumR5, cumR10, cumR20, cumRall))

plt.figure()
plt.boxplot(data)
plt.grid('on')
plt.title('Objective Function Variablility for Number of Action Change')
plt.xticks([1,2,3,4,5], ('4','5','10','20', 'all'))
plt.xlabel('no. of action changes in total 100 timesteps')
plt.ylabel('cumulative reward')
plt.show()