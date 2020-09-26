#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv


# In[2]:


cost_HC = input('Please input the cost for hiring one celebrity (cost_HC): ')
cost_SMA = input('Please input the cost to increase the budget for social media advertersiing by 1% (cost_SMA): ')
cost_TM = input('Please input the cost to increase the number of paid position in telemdeicine by 1% (cost_TM): ')
cost_AT = input('Please input the cost to create one piece of art therapy (cost_TM): ')
cost_SAA = input('Please input the cost to increse social awareness activity by 1% (cost_SAA): ')
cost_SG = input('Please input the cost to increase the number of online support groups by 1% (cost_SG): ')


# In[3]:


costs = {'HC':float(cost_HC), 'SMA':float(cost_SMA), 'TM':float(cost_TM), 'AT':float(cost_AT), 'SAA':float(cost_SAA), 'SG':float(cost_SG)}


# In[4]:


costs_sorted = sorted(costs.items(), key=lambda x: x[1], reverse=True)


# In[5]:


states = ['Red','Yellow','Blue']


# In[6]:


# TP algorithm to create transition probability matrix based on cost ordering

transitions = []

np.random.seed(99)
prob = np.random.dirichlet(np.ones(len(costs_sorted)),size=1)[0].tolist()
prob_asc = sorted(prob)
prob_eq = (np.ones(len(costs_sorted))/len(costs_sorted)).tolist()
sum_list = [a + b for a, b in zip(prob_asc, prob_eq)]
prob_des = [1 - number for number in sum_list]

for i in range(len(states)):
    for j in range(len(states)):
        for k in range(len(costs_sorted)):
            if ((states[i] == 'Red') and (states[j] == 'Red')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_asc[k]])
            elif ((states[i] == 'Red') and (states[j] == 'Yellow')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_eq[k]])
            elif ((states[i] == 'Red') and (states[j] == 'Blue')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_des[k]])
            elif ((states[i] == 'Yellow') and (states[j] == 'Red')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_asc[k]-0.002])
            elif ((states[i] == 'Yellow') and (states[j] == 'Yellow')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_eq[k]])
            elif ((states[i] == 'Yellow') and (states[j] == 'Blue')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_des[k]+0.002])
            elif ((states[i] == 'Blue') and (states[j] == 'Red')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_asc[k]-0.004])
            elif ((states[i] == 'Blue') and (states[j] == 'Yellow')):
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_eq[k]])
            else:
                transitions.append([states[i], costs_sorted[k][0], states[j], prob_des[k]+0.004])
                
        
    


# In[7]:


cols = ['From_State','Action','To_State','Probability']
df_transitions = pd.DataFrame(transitions,columns=cols)
df_transitions.head()


# In[8]:


df_transitions.to_csv('transitions.csv',index=False, header=False)


# In[9]:


# Problm Parameters
gamma = 0.8;
epsilon = 0.001;

    
# Read the probability transition data (S*S*A), and rewad data (S*A)
Transitions = {}; # dictionary
Reward = {}; # dictionary

with open('transitions.csv', 'r') as csvfile:
    #reader = pd.read_csv(csvfile,delimiter=',');
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] in Transitions:
            if row[1] in Transitions[row[0]]:
                Transitions[row[0]][row[1]].append((float(row[3]), row[2]))
            else:
                Transitions[row[0]][row[1]] = [(float(row[3]), row[2])]
        else:
            Transitions[row[0]] = {row[1]:[(float(row[3]),row[2])]}

    #read rewards file and save it to a variable
with open('rewards.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Reward[row[0]] = float(row[1]) if row[1] != 'None' else None  
#%% Value iteration: Main loop
"""
Solving the MDP by value iteration.
returns utility values for states after convergence
"""
states =  Transitions.keys();
#actions = mdp.actions
#print(states); print(actions);
#initialize value of all the states to 0 (this is k=0 case)
V1 = {s: 0 for s in states}
while True:
    V = V1.copy()
    delta = 0
    for s in states:
        #Bellman update, update the utility values
        V1[s] = Reward[s] + gamma * max([ sum([p * V[s1] for (p, s1)
        in Transitions[s][a]]) for a in Transitions[s].keys()]);
        #calculate maximum difference in value
        delta = max(delta, abs(V1[s] - V[s]))

    #check for convergence, if values converged then return V
    if delta < epsilon * (1 - gamma) / gamma:
        break;
        
        
#%% Post solution analysis   
pi = {}
for s in states:
    pi[s] = max(Transitions[s], key=lambda a: sum([p * V[s1] for (p, s1) in Transitions[s][a]])); 
       
print( 'State - Value')
for s in V:
    print( s, ' - ' , V[s]);
print ('\nOptimal policy is \nState - Action')
for s in pi:
    print( s, ' - ' , pi[s]);


# In[ ]:




