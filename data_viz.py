import numpy as np
import matplotlib.pyplot as plt

ENV = 'CartPole-v0'
CASE_1_TITLE = 'with target network'
CASE_2_TITLE = 'withour target network'

CASE_1_PATH = 'with_target_results/'
CASE_2_PATH = 'without_target_results/'
CASE_1_CSV_INITIALS = 'expt'
CASE_2_CSV_INITIALS = 'expt'

file_path_1 = CASE_1_PATH + CASE_1_CSV_INITIALS
file_path_2 = CASE_2_PATH + CASE_2_CSV_INITIALS

openAI_list = []
in_house_list = []

n_expmt = 15

for i in range(n_expmt):
    # in_house_list.append(np.load('in_house_code_results/'+str(version)+'/expt'+str(i)+'.npy'))
    in_house_list.append(np.loadtxt(file_path_1+str(i)+'.csv', delimiter=',', skiprows=1))
    openAI_list.append(np.loadtxt(file_path_2+str(i)+'.csv', delimiter=',', skiprows=1))

# find minimum episode experiment
min_epsd = 10000
for i in range(n_expmt):
    lb = np.min ( [ np.min( in_house_list[i].shape[0]) ,  np.min(openAI_list[i].shape[0] ) ] )
    if lb < min_epsd:
        min_epsd = lb


n = len(openAI_list)
interval = 100

y_openAI = []
y_in_house = []
for i in range(n):
    y_openAI.append(openAI_list[i][:min_epsd,2])
    y_in_house.append(in_house_list[i][:min_epsd,2])

y_openAI = np.array(y_openAI)
y_in_house = np.array(y_in_house)

eps_openAI = []
eps_in_house = []
for i in range(n):
    eps_openAI.append(openAI_list[i][:min_epsd,0])
    eps_in_house.append(in_house_list[i][:min_epsd,0])

eps_openAI = np.array(eps_openAI)
eps_in_house = np.array(eps_in_house)

# data vizualization : comparison between openAI and DQN vanilla code
plt.clf()
x = in_house_list[0][:min_epsd, 1]
plt.plot(x, np.nanmedian(y_openAI, axis=0) )
plt.fill_between(x, np.nanpercentile(y_openAI, 25, axis=0), np.nanpercentile(y_openAI, 75, axis=0), alpha=0.25)
plt.plot(x, np.nanmedian(y_in_house, axis=0) )
plt.fill_between(x, np.nanpercentile(y_in_house, 25, axis=0), np.nanpercentile(y_in_house, 75, axis=0), alpha=0.25)
plt.grid('True')
plt.legend(['openAI baseline code', 'DQN vanilla'])
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('MountainCar-v0 case comparison \n computation time per expmt \n openAI baseline (~1 h 12 mins); DQN vanilla (~10 mins)')
plt.savefig('plots/'+str(version)+'/comparison.png')

# data vizualization : exploration for episodes in OpenAI experiments
plt.clf()
x = in_house_list[0][:min_epsd, 1]
plt.plot(x, np.nanmedian(y_openAI, axis=0) )
plt.fill_between(x, np.nanpercentile(y_openAI, 25, axis=0), np.nanpercentile(y_openAI, 75, axis=0), alpha=0.25)
plt.grid('True')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Exploration in openAI experiments')
ax2 = plt.twinx()
plt.plot(x, np.nanmedian(eps_openAI, axis=0), color='r' )
plt.fill_between(x, np.nanpercentile(eps_openAI, 25, axis=0), np.nanpercentile(eps_openAI, 75, axis=0), alpha=0.25, color='r')
plt.ylabel('% time spent exploring')
plt.savefig('plots/'+str(version)+'/open_AI_baseline.png')


# data vizualization : exploration for episodes in vanilla DQN experiments
plt.clf()
x = in_house_list[0][:min_epsd, 1]
plt.plot(x, np.nanmedian(y_in_house, axis=0) )
plt.fill_between(x, np.nanpercentile(y_in_house, 25, axis=0), np.nanpercentile(y_in_house, 75, axis=0), alpha=0.25)
plt.grid('True')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Exploration in vanilla DQN experiments')
ax2 = plt.twinx()
plt.plot(x, np.nanmedian(eps_in_house, axis=0), color='r' )
plt.fill_between(x, np.nanpercentile(eps_in_house, 25, axis=0), np.nanpercentile(eps_in_house, 75, axis=0), alpha=0.25, color='r')
plt.ylabel('% time spent exploring')
plt.savefig('plots/'+str(version)+'/vanilla_dqn.png')

