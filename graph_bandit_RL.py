import gym
import numpy as np
import pandas as pd
from known_rewards_helper_functions import get_Q_table, all_paths
from estimator import NormalBayesianEstimator, AverageEstimator
import random
from scipy.stats import norm
import networkx as nx

class GraphBandit(gym.Env):
    """
    The graph bandit implemented as a gym environment
    """
    
    def __init__(self, mean, stddev, G, belief_update=None, bayesian_params=[0, 1, 1], init_state=3, uncertainty=1,\
                 Q_table_version=None, local_sampling=None, time_varying=False, eta=None,\
                 a=None, N=100):
        """
        param mean: Vector of means for bandits (one mean for each node in graph G).
        param G: networkx Graph.
        param stddev: Vector of stddevs for bandits (one stddev for each node in graph G).
        param G: networkx graph for graph setup.
        param belief_update: How to update beliefs (when applicable (in Q-learning algorithms)). None=use sample rewards in QL-                                algorithms, average=use average, Bayesian=use Bayesian update. 
        param bayesian_params: [mu0, sigma0^2, sigma^2]; a list of the belief parameters. mu0 is the initial mean estimate, sigma0                               is the initial standard deviation in the initial belief, and sigma is the known variance of the                                   reward distributions. All parameters are the same for all nodes.
        param init_state: the initial state (node) of the agent.
        param uncertainty: Constant in UCB estimate; e.g. mu_UCB = mu_est + uncertainty/sqrt(t)
        Q_table_version: Only applicable if using Q_table + UCB or Q_table + Thompson sampling (otherwise let it be None)
                 UCB for Q_table + UCB and Thompson for Q_table + Thompson sampling
        param local_sampling: local sampling algorithms; local_Thompson, local_UCB, or local_greedy 
        """
        # Initialization parameters
        self.mean = mean.copy()
        self.stddev = stddev.copy()
        self.G = G.copy()
        self.belief_update = belief_update
        self.bayesian_params = bayesian_params.copy()
        self.state = init_state
        self.uncertainty = uncertainty
        self.Q_table_version = Q_table_version
        self.local_sampling = local_sampling
        self.nodes = self.G.nodes
        self.edges = self.G.edges
        self.neighbors = [list(self.G.neighbors(node)) for node in self.nodes]
        self.mu_best = np.argmax(self.mean)
        self.all_states = []
        self.all_rewards = {i:[0] for i in range(len(self.mean))}
        self.all_maxes = []
        self.visited_expected_rewards = []
        
        # Number of nodes
        self.num_nodes = self.mean.shape[0]
        
        # Rewards collected at each node during training
        self.collected_rewards = {i: [] for i in range(self.num_nodes)}
        
        self.time_varying = time_varying
        if self.time_varying:
            assert (eta is not None) and (a is not None) and (N is not None)
            self.eta = eta
            self.a = a
            self.N_memory = N
            self.c = self.mean / eta
            
        
        
        

        
        
        # Initialize Q-table
        # TODO: For now if/else are the same. Change to incorporate initial belief in UCB/Thompson
        if self.Q_table_version is None:
            self.q_table = np.zeros((self.num_nodes,self.num_nodes))            
        else:
            self.q_table = np.zeros((self.num_nodes,self.num_nodes))
                 
        # Eliminate actions that are 'illegal' (agent can only go via edges)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if (i,j) not in G.edges:
                    self.q_table[i,j] = -np.inf
                    
        # Store history of rewards and number of vistits for each node, as well as number of (state -> action) encounters
        for n in self.nodes:
            self.nodes[n]['r_hist']=[] 
            self.nodes[n]['n_visits']=0 
            self.nodes[n]['actions'] = {(n,n): 0}
            for m in self.neighbors[n]:
                self.nodes[n]['actions'][(n,m)] = 0 # State action count
                
        
        # Total number of visits per node (nodes are ordered by zero-indexing)
        self.visits = np.array([0 for _ in range(self.num_nodes)])  # Another way to access visits (as list) for faster computation
            
        # Initialize estimators if applicable (either Bayesian estimators, or average estimators)
        if self.belief_update=='Bayesian' or self.belief_update=='Bayesian_full_update' or \
        self.Q_table_version=='Thompson' or self.Q_table_version=='UCB' or self.local_sampling is not None:
            assert bayesian_params is not None
            # Confidence bounds
            self.upper95 = {node: bayesian_params[0] + self.uncertainty*np.sqrt(bayesian_params[1]) for node in self.nodes}
            self.lower95 = {node: bayesian_params[0] - self.uncertainty*np.sqrt(bayesian_params[1]) for node in self.nodes}
            # Bayesian estimators for each node's reward distribution
            for n in self.nodes:
                self.nodes[n]['est']= NormalBayesianEstimator(bayesian_params[0], bayesian_params[1], bayesian_params[2])
        elif self.belief_update=='average' or self.belief_update=='average_full_update':
            # Average estimators for each node's reward distribution
            for n in self.nodes:
                self.nodes[n]['est']= AverageEstimator()
                
        # List of all visited states
        self.visitedStates = [] # Ignore initial state since no reward is received here (TODO: Maybe change this)
        
        # List of nodes to not visit (if applicable)
        self.do_not_visit = [False for _ in range(len(G.nodes))]
        
        # Has Q-table been generated (if applicable)
        self.q_table_generated = False
        
        
    def step(self, action, update_Q=False, gamma=0.1):
        # Take a step in the graph with 
        # Returns: observation, QL_reward, done
        assert action <= self.num_nodes
        
        if (action, self.state) in self.G.edges:
#             alph = 8
            reward = np.random.uniform(low = self.mean[action]-0.5, high=self.mean[action]+0.5)
#             reward = np.random.beta(a=alph, b = alph/self.mean[action]-alph)
            
#             reward = np.random.normal(loc=self.mean[action], scale=1)
#             reward = max(reward, 0)
#             reward = min(reward, 1)
#             print(reward)
            

            

    
#             reward = np.random.uniform(low=0, high = 2*self.mean[action])
                
#             print('1', reward)
#             reward = min(reward, 1)
#             reward = max(reward, 0)
#             print('2', reward)

#             reward = np.random.uniform(low=0, high=self.mean[action])
            
            state_old = self.state
            self.state = action
            
            
                        
            self.nodes[action]['r_hist'].append(reward)
            
            self.visits[self.state] += 1
            self.nodes[self.state]['n_visits'] += 1
            self.all_maxes.append(np.max(self.mean)) 
            self.visited_expected_rewards.append(self.mean[self.state])
            self.all_rewards[self.state].append(reward) 
            
            # Determine what R (reward) to use in Q-learning algorithm.
            # Sampled reward or full estimate.
            try:
                self.nodes[action]['est'].update(reward)
                QL_reward = self.nodes[action]['est'].get_param()[0]
            except:
                QL_reward = reward
                
        else:
            print(self.state)
            print(action)
            QL_reward = -100
            done = True
            print("Something is wrong. Illegal action")
            
        self.visitedStates.append(self.state)
        if update_Q:
            self.update_q_table(state_old, action, gamma)
        return self.state, QL_reward, False
    
    def show_q_table(self):
        """
        Prints the Q-table.
        """
        print(pd.DataFrame(self.q_table))
    
    def reset(self, node):
        """
        Resets agent at node.
        If node=None, random node is picked.
        """
        if node is None:
            self.state = np.random.randint(self.num_nodes)
        else:
            self.state = node
        return self.state
    
    def explore(self, node):
        """
        Returns least explored available action.
        """
        
        neighbors = self.neighbors[node]
        N_explorations = self.visits[neighbors]
        return neighbors[np.argmin(N_explorations)]

    def visit_all_nodes_old(self, node):
        """
        Visits all nodes in graph (at least) once.
        
        param node: current state.
        """            
        # Explore neighboring nodes
        for neighbor in list(self.neighbors[node]):
            if neighbor != node:
                if self.visits[neighbor] == 0:
                    return neighbor
                    
        # All neighboring nodes explored
        neighbors = list(self.neighbors[node])
        neighbors.remove(node)
        neighbors=np.array(neighbors)
        return 

        print('failed')
  
    def explore_efficiently(self, node, N=None):
        """
        TODO: Needs some fixing.
        
        Efficient exploration algorithm.
        
        param node: current state
        param N: number of steps to explore
        """
        if N is not None:
            self.N_actual = N
            self.N = N
            self.K = round(self.N / len(self.nodes)) # Number of times to explore each node
            self.do_not_visit = [False for _ in range(len(self.nodes))]
            
        # If exploration is done (take greedy action)
        if np.sum(self.visits) >= self.N_actual:
            return np.argmax(self.q_table[node])
        
        # Not done exploring current node not 
        if self.visits[node] < self.K:
            return node
        
        # Explore neighboring nodes
        for neighbor in list(self.neighbors[node]):
            if neighbor != node:
                if self.visits[neighbor] < self.K:
                    return neighbor
                    
        # All neighboring nodes explored
        for neighbor in list(self.neighbors[node]):
            if neighbor != node:
                if not self.do_not_visit[neighbor]:
                    self.do_not_visit[neighbor] = True
                    self.N = self.N-1
                    self.K = round(self.N / len(self.nodes))
                    return neighbor
        print('failed')
    
    def QL_alg1_action(self, node):
        """
        TODO: Probably the same as function explore()
        
        epsilon-greedy algorithm where 'epsilon-step' chooses least visited node
        """
        action = node
        min_exploration = self.visits[node]
        for neighbor in self.neighbors[node]:
            if self.visits[neighbor] < min_exploration:
                action = neighbor
                min_exploration = self.visits[neighbor]
        return action
    
    def visit_all_nodes(self, update_Q=False, gamma=0.1):
        while True:
            unvisited = [i for i in range(self.num_nodes) if self.visits[i]==0]
            if len(unvisited)==0:
                break

            dest = unvisited[0]

            next_path = nx.shortest_path(self.G,self.state,dest)
            
            if len(next_path) == 1 and next_path[0] == self.state:
                self.step(self.state, update_Q, gamma)
            else:
                for s in next_path[1:]:
                    self.step(s, update_Q, gamma)
    
    def local_thompson_sampling(self, node):
        """
        Local Thompson sampling
        """
        # Deciding next s using Thompson Sampling.

        zs = list(self.neighbors[node])
        params = np.array([list(self.nodes[node]['est'].get_param()) for node in zs])
#         samples = np.random.normal(loc=params[:,0], scale=np.sqrt(params[:,1]))
        samples = np.random.beta(a=params[:,0]+1, b=params[:,1]+1)
        
        
        z_star = zs[np.argmax(samples)]

        return z_star
    
    def local_greedy(self, node, epsilon):
        """
        Local greedy action
        """
        # Deciding next s using Thompson Sampling.
        muhats = []
        zs = []
#         epsilon = epsilon0 / (np.min(self.visits)+1)

        if np.random.rand() < epsilon:
            z_star = self.explore(node)
        else:
            zs = list(self.neighbors[node])
            params = np.array([list(self.nodes[node]['est'].get_param()) for node in zs])
            z_star = zs[np.argmax(params[:,0])]

        return z_star
    
    def local_UCB(self, node, h):
        """
        Local UCB
        """
        
        zs = self.neighbors[node]        
#         params = np.array([list(self.nodes[node]['est'].get_param()) for node in zs])
#         samples = params[:,0] + self.uncertainty * np.sqrt(np.log(h+1)/(self.visits[zs] + 1))

        ts = np.array([self.visits[z] for z in zs])
        
        means = np.array([np.mean(self.all_rewards[i]) for i in range(self.num_nodes)])
        
        h = np.sum(self.visits)
        
        samples = means[zs] + np.sqrt(2*np.log(h+1)/ts)

        z_star = zs[np.argmax(samples)]

        return z_star
    
    def QL_alg4_action(self, node):
        """
        Local UCB with elimination
        """
        muhats = []
        zs = []
        for z in list(self.G[node])+[node]:
            
            if self.q_table[node, z] > -np.inf: # Do not consider eliminated state-action pairs
                
                mu_1,var_1 = self.nodes[z]['est'].get_param()

                muhats.append(mu_1+self.uncertainty * np.sqrt(var_1))
                zs.append(z)
            else:
                muhats.append(-np.inf)
                zs.append(z)
                
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def global_UCB(self):
        
        zs = list(self.nodes)
        ts = np.array([self.visits[z] + 1 for z in zs])
        
        params = np.array([list(self.nodes[node]['est'].get_param()) for node in zs])
        means = params[:,0]
#         means = np.array([np.mean(self.all_rewards[i]) for i in range(self.num_nodes)])
        
        h = np.sum(self.visits)
        
        samples = means + np.sqrt(2*np.log(h+1)/ts)
        
        return samples
    
    def global_thompson_sampling(self):
        """
        Samples from posterior distributions at each node in the whole graph.
        """

        zs = list(self.nodes)
        params = np.array([list(self.nodes[node]['est'].get_param()) for node in zs])
        samples = np.random.normal(loc=params[:,0], scale=np.sqrt(params[:,1]))
        
        return samples

    
    def get_bayesian_estimates(self):
        mus = []
        sigs = []
        for node in self.nodes:
            mu, var = self.nodes[node]['est'].get_param()
            mus.append(mu)
            sigs.append(sigs)
        return mus, sigs
    
    def compute_success(self):
        """
        returns 1 if current best node belief is right; returns 0 otherwise
        """
        try: # Check Bayesian estimates
            estimates = np.array([np.mean(self.all_rewards[i]) for i in range(self.num_nodes)])
#             params = np.array([list(self.nodes[node]['est'].get_param()) for node in self.nodes])
#             best_node = np.argmax(params[:,0])
            best_node = np.argmax(estimates)
        except: # Try Q-table (for Q-learning methods)
            best_node = np.argmax(np.diagonal(self.q_table))
        if best_node == self.mu_best:
            return 1
        else:
            return 0
    
    def iota(self, k, T):
        """iota in UCB-hoeffding algorithm"""
        return np.log(self.num_nodes**2 * T *(k+1)*(k+2))
    
    def update_means(self, state, reward):
        """Updates in time varying mean case"""
        if len(self.all_states) >= self.N_memory: # Only keep last N in memory
            to_remove = self.all_states[0]
            self.all_states = self.all_states[1:].copy()
            self.visits[to_remove] -= 1
            self.all_rewards[to_remove] = self.all_rewards[to_remove][1:].copy()
        self.c = self.c + self.a
        self.c = np.array([min(self.c[i],100) for i in range(self.num_nodes)])
        self.c[state] = max(self.c[state] - reward, 0)
        self.mean = self.eta * self.c
        
        
        
    def train_agent(self, episodes = 1, H=100, init_node= None, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_discount=1,
                    T_eps_max=np.inf, QL_type=0, update_multiple_qs=False, efficient_explore_length=None,\
                   start_with_exploration=False, update_frequency=None, random_path=False):
        """
        Training the agent
        
        param episodes: number of episodes to train
        param H: number of steps per episode
        param init_node: initial node 
        param alpha: Q-learning parameter (if applicable)
        param gamma: RL discount factor (if applicable)
        param epsilon: exploration parameter (if applicable)
        param epsilon_discount: Discount parameter for epsilon parameter.
        param T_eps_max: Stop exploring after this many time steps (each episode).
        param QL_type: integer; 0=standard eps-Greedy; 1=local exploration; 2,3,4,5,8
        param update_multiple_qs: if True, multiple state-action pairs are updated each step, by exploiting
                                  rewards R(state,action) independance of the 'state' part
        param efficient_explore_length: param N in explore_efficiently() function (if applicable) 
        param start_with_exploration: if True and self.Q_table_version is UCB or Thompson, then agent will first visit
        each node once to get an initial estimate
        param update_frequency: How many time steps between Q-table update (if applicable (when self.Q_table_version is UCB or                                     Thompson))
        """
        epsilon0 = epsilon
        if update_frequency is None:
            update_frequency = self.num_nodes
            
            
            
        Deltas = np.max(self.mean) - self.mean
        Delta_min = np.min(Deltas[np.nonzero(Deltas)])
        
        K = episodes
        H_hoeffding = np.log(2/(1-gamma)/(Delta_min))/np.log(1/gamma)
        
        self.success = []
        
        if self.Q_table_version == 'hoeffding': # UCB-hoeffding approach, efficient Q-learning
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if (i,j) in self.edges:
                        self.q_table[i,j] = 1/(1-gamma)
                 
        self.Q_hat = self.q_table.copy() 
                    

        

        for k in range(1, episodes+1):
            state = self.reset(init_node)

            for h in range(0, H):


#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Determine action--------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
                # Reduce epsilon
        
#                 epsilon = epsilon0 / (np.sum(self.visits)+1)
                # epsilon = epsilon0*epsilon_discount**h
                epsilon = 0.7*(3*self.num_nodes + 1) / (3*self.num_nodes + h)
                if QL_type is not None:
                    if QL_type == 1: # 'Greedy efficient' exploration
                        if False: # len(self.visitedStates) == 0:
                            self.visit_all_nodes(update_Q=True, gamma=gamma)
                            action = None
                        else:
                            if random.uniform(0, 1) < epsilon:
                                action = self.explore(state)
                            else:
                                action = np.argmax(self.q_table[state]) # Exploit learned values


                    elif QL_type ==0: # Epsilon greedy
                        if random.uniform(0, 1) < epsilon and h<T_eps_max:
                                action = random.choice(list(self.neighbors[state]))  
                        else:
                            action = np.argmax(self.q_table[state]) # Exploit learned values

                    elif QL_type == 2:
                        if random.uniform(0, 1) < self.epsilons[state] and h<T_eps_max:
                            action = self.local_thompson_sampling(state)
                        else:
                            action = np.argmax(self.q_table[state]) # Exploit learned values
                    elif QL_type == 3:
                        if random.uniform(0, 1) < self.epsilons[state] and h<T_eps_max:
                            action = self.local_UCB(state, h)
                        else:
                            action = np.argmax(self.q_table[state]) # Exploit learned values
                    elif QL_type == 4:
                            if random.uniform(0, 1) < self.epsilons[state] and h<T_eps_max:
                                action = self.QL_alg4_action(state)
                            else:
                                action = np.argmax(self.q_table[state]) # Exploit learned values
                    elif QL_type == 5:
                        assert efficient_explore_length is not None
                        if h==0:
                            N=efficient_explore_length
                        else:
                            N=None
                        action = self.explore_efficiently(state, N=N) 
                    elif QL_type == 8:
                        assert efficient_explore_length is not None
                        if h==0:
                            N=efficient_explore_length
                        else:
                            N=None
                        if h==efficient_explore_length:
                            means = np.array([self.nodes[i]['est'].get_param()[0] for i in self.nodes])
                            self.q_table = get_Q_table(self.G, means, T=rounds-efficient_explore_length)
                        action = self.explore_efficiently(state, N=N) 
                    elif QL_type == 'Thompson':  
                        if h == 100:
                            # action = self.explore_efficiently(state, N=20) 
                            means = np.array([self.nodes[i]['est'].get_param()[0] for i in self.nodes])
                            self.q_table = get_Q_table(self.G, means, T=rounds-efficient_explore_length)
                            action = np.argmax(self.q_table[state])
                        elif h > 100:
                            action = np.argmax(self.q_table[state])
                        else:
                            if (np.array([i+h for i in range(3)]) % 10 == 0).any():
                                lower_bound = self.lower95[state]
                                upper_bounds = list(self.upper95.values())
                                upper_bounds[state] = -np.inf
                                upper_bound = np.max(upper_bounds)
                                if (upper_bound > lower_bound): # or np.min(self.visits) < 1) and j<100:
                                    action = self.explore(state)
                                else:
                                    action = self.local_thompson_sampling(state)
                            else:
                                action = self.local_thompson_sampling(state)
                    
                # Q-table+UCB or Q-table+Thompson sampling
                elif QL_type is None:
                    if self.Q_table_version is not None:
                        if self.Q_table_version == 'hoeffding': # UCB-hoeffding, efficient Q-learing
                            action = np.argmax(self.q_table[state])

                        else:  # The propsed Q-graph approach in the paper
                            if (h%update_frequency==0):
                                if self.Q_table_version == 'UCB':
                                    means = self.global_UCB()

                                elif self.Q_table_version=='Thompson':
                                    means = self.global_thompson_sampling()

                                # Generate "known rewards" Q-table with mean estimates
                                if not random_path:
                                    self.q_table,_ , _ = get_Q_table(self.G, means, self.num_nodes)
                                else:
                                    highest_sample = np.argmax(means)
                                    all_paths_temp = all_paths(self.G, state, highest_sample)
                                    if len(all_paths_temp) > 1:
                                        path = np.random.choice(all_paths_temp)
                                    else:
                                        path = all_paths_temp[0]
                                    path.pop(0) # Remove current node
                                    while len(path) > update_frequency:
                                        path = np.random.choice(all_paths_temp)
                                        path.pop(0) # Remove current node
                                    path = path + [highest_sample for _ in range(update_frequency-len(path))]
                            if not random_path:
                                action = np.argmax(self.q_table[state])
                            else:
                                action = path[h%update_frequency]
                           
                            
                    elif self.local_sampling is not None:
                        if self.local_sampling == 'local_Thompson':
                            if len(self.visitedStates) == 0:
                                self.visit_all_nodes()
                                action = None
                            else:
                                action = self.local_thompson_sampling(state)
                        elif self.local_sampling == 'local_UCB':
                            if len(self.visitedStates) == 0:
                                self.visit_all_nodes()
                                action = None
                            else:
                                action = self.local_UCB(state, h)
                        elif self.local_sampling == 'local_greedy':
                            action = self.local_greedy(state, epsilon)
#-------------------------------------------------------------------------------------#                      
#-------------------------------------------------------------------------------------#
#-------------------------------End of determine action-------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################

                # Next state, reward, and store reward
                if action is not None:
                    next_state, QL_reward, done = self.step(action)
            
            
            
        
        
        
#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Update Q-table (if applicable)------------------------#
#---------NOTE: This section is skipped for Q-table + UCB/Thompson approach-----------#
#-------------------------------------------------------------------------------------#
                
                if self.Q_table_version == 'hoeffding': # UCB-hoeffding, efficient Q-learing
                    k = self.visits[action] + 1
                    c_2 = 4*np.sqrt(2)
                    iota = self.iota(k, H)
                    b_k = c_2/(1-gamma) * np.sqrt(H_hoeffding*iota/k)
                    alpha_k = (H_hoeffding+1)/(H_hoeffding+k)
                    
                    V_hat = np.max(self.Q_hat[action])
                    for neighbor in self.neighbors[action]:
                        self.q_table[neighbor, action] = (1-alpha_k)*self.q_table[neighbor, action] \
                            + alpha_k*(QL_reward + b_k + gamma*V_hat)
                        self.Q_hat[neighbor, action] = min(self.q_table[neighbor, action], self.Q_hat[neighbor, action])

                    
                        
                        
                        
        
        


   

                if QL_type is not None and action is not None:
                    self.update_q_table(state, action, gamma)

#                         self.nodes[node]['actions'][(node, next_state)] += 1 # Update state-action count



#                     mu_temp, sigma_temp = self.nodes[action]['est'].get_param()
#                     # Update upper and lower confidence bounds
#                     self.upper95[action] = mu_temp + self.uncertainty*sigma_temp
#                     self.lower95[action] = mu_temp - self.uncertainty*sigma_temp
#                     if  self.upper95[action] < self.lower95[action]:
#                         print('Something is wrong. UCB<LCB')

#                     if QL_type == 4: # Eliminate high-confidence "bad" choice
#                         lower_confidence = list(self.lower95.values())
#                         lower_confidence = np.array(lower_confidence)
#                         if (self.upper95[action] < lower_confidence[list(self.neighbors[action])]).any():
#                             self.q_table[state, action] = -np.inf
                            
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------End of update Q-table---------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################



#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Update parameters-------------------------------------#
                
                state = self.state
#                 self.visits[state] += 1 in self.step() function
                self.success.append(self.compute_success())
                self.all_states.append(state)
#                 self.all_rewards[state].append(QL_reward) in self.step() function
#                 self.all_maxes.append(np.max(self.mean)) in self.step() function
#                 self.visited_expected_rewards.append(self.mean[state])
                
                if self.time_varying:
                    self.update_means(state, QL_reward)

                    



#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################
    
    def update_q_table(self, state, action, gamma):
        next_state = action
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

    #                     if self.belief_update!='Bayesian_full_update' and self.belief_update!='average_full_update':
    #                         new_value = (1 - alpha) * old_value + alpha * (QL_reward + gamma * next_max)

    #                     else:
    #                         new_value = self.nodes[action]['est'].get_param()[0] + gamma * next_max # Full Bayesian update


        for node in self.neighbors[action]:
            old_value = self.q_table[node, action]
            if self.belief_update!='Bayesian_full_update' and self.belief_update!='average_full_update':
                new_value = (1 - alpha) * old_value + alpha * (QL_reward + gamma * next_max)
            else: 
#                 new_value = self.nodes[action]['est'].get_param()[0] + gamma * next_max # Full Bayesian update
                new_value = np.mean(self.all_rewards[action]) + gamma * next_max
            self.q_table[node, action] = new_value
                
    def runAgent(self, T=100, state=None):
        """
        NOTE: Run train_agent first!
        Runs agent for T time steps using greedy Q-table actions.
        
        param T: Length of game.
        param state: Initial state.
        """
         # For plotting metrics
        all_states = []
        all_rewards = []
        all_regrets = []
        reward = 0
        if state is None:
            state = self.reset()
        self.state = state
        for i in range(T):

            all_states.append(state)

            action = np.argmax(self.q_table[state]) # Exploit learned values

            next_state, reward, done = self.step(action) 
            all_rewards.append(reward)
            all_regrets.append(np.max(self.mean)-self.mean[action])


            state = next_state

        return all_states, all_rewards, all_regrets
    
    def expectedRegret(self):
        # Returns vector of expected regret
        mu_max = np.max(self.mean)
        mu_visited = self.mean[self.visitedStates]
        
        return np.array(self.all_maxes) - np.array(self.visited_expected_rewards)