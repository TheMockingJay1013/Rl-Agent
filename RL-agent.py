import gridworld
import numpy as np
import random
import matplotlib.pyplot as plt 
import cv2
episode_num=2000
steps=500


discount_rate=1
learning_rate=0.1

exploration_rate=1
max_exploration_rate=1
exploration_decrement=0.01


reward_all_episode=[]
world = gridworld.GridWorld()

q_value=np.zeros([10,15,4],dtype=float)
print(q_value)

for epsiode in range(episode_num):
    state=world.START
    reward_current_episode=0
    
    for step in range(steps):
        x=random.uniform(0,1)
        if x>exploration_rate:
            action=np.argmax(q_value[state[0],state[1],:])
        else:
            action=random.randint(0,3)
        
        new_state,reward=world.step(state,action)
        
        
        q_value[state[0],state[1],action]=q_value[state[0],state[1],action]*(1-learning_rate) + learning_rate*(reward + discount_rate*np.max(q_value[new_state[0],new_state[1],:]))
        
        state=new_state
        reward_current_episode+=reward
        if reward==0:
            break
    
    exploration_rate=(max_exploration_rate)*np.exp(-exploration_decrement*epsiode)
    
    reward_all_episode.append(reward_current_episode)
    
    
x=[i for i in range(2000)]  

plt.bar(x,reward_all_episode)
plt.show()  

img=cv2.imread('grid.png')
dimension=img.shape

hori_unit=int(dimension[1]/16)
verti_unit=int(dimension[0]/11)
origin=[verti_unit*3//2,hori_unit*3//2]


state=world.START
trial_reward=0
print(origin[0]+verti_unit*state[0],origin[1]+ hori_unit*state[1])
while(True):
    print(state)
    action=np.argmax(q_value[state[0],state[1],:])
    new_state,reward=world.step(state,action)
    cv2.line(img,(int(origin[1]+hori_unit*state[1]),int(origin[0]+ verti_unit*state[0])),(int(origin[1]+hori_unit*new_state[1]),int(origin[0]+ verti_unit*new_state[0])),(255,0,0),4)
    state=new_state
    trial_reward+=reward
    if reward==0:
        print(state)
        print(trial_reward)
        break
cv2.imshow("grid",img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit()
    
        
