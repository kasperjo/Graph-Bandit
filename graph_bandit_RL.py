import gym
import numpy as np
import pandas as pd
from known_rewards_helper_functions import get_Q_table
from estimator import NormalBayesianEstimator, AverageEstimator
import random

class GraphBandit(gym.Env):
    """
    The graph bandit implemented as a gym environment
    """
    
    def __init__(self, mean, stddev, G, belief_update=None, bayesian_params=[0, 1, 1], init_state=3, uncertainty=1,\
                 Q_table_version=None, local_sampling=None):
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
        
        # Number of nodes
        self.num_nodes = self.mean.shape[0]
        
        # Rewards collected at each node during training
        self.collected_rewards = {i: [] for i in range(self.num_nodes)}
        
        
        

        
        
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
        for n in self.G.nodes:
            self.G.nodes[n]['r_hist']=[] 
            self.G.nodes[n]['n_visits']=0 
            self.G.nodes[n]['actions'] = {(n,n): 0}
            for m in self.G.neighbors(n):
                self.G.nodes[n]['actions'][(n,m)] = 0 # State action count
                
        
        # Total number of visits per node (nodes are ordered by zero-indexing)
        self.visits = [0 for _ in range(self.num_nodes)]  # Another way to access visits (as list) for faster computation
            
        # Initialize estimators if applicable (either Bayesian estimators, or average estimators)
        if self.belief_update=='Bayesian' or self.belief_update=='Bayesian_full_update' or \
        self.Q_table_version=='Thompson' or self.Q_table_version=='UCB' or self.local_sampling is not None:
            assert bayesian_params is not None
            # Confidence bounds
            self.upper95 = {node: bayesian_params[0] + self.uncertainty*np.sqrt(bayesian_params[1]) for node in self.G.nodes}
            self.lower95 = {node: bayesian_params[0] - self.uncertainty*np.sqrt(bayesian_params[1]) for node in self.G.nodes}
            # Bayesian estimators for each node's reward distribution
            for n in self.G.nodes:
                self.G.nodes[n]['est']= NormalBayesianEstimator(bayesian_params[0], bayesian_params[1], bayesian_params[2])
        elif self.belief_update=='average' or self.belief_update=='average_full_update':
            # Average estimators for each node's reward distribution
            for n in self.G.nodes:
                self.G.nodes[n]['est']= AverageEstimator()
                
        # List of all visited states
        self.visitedStates = [] # Ignore initial state since no reward is received here (TODO: Maybe change this)
        
        # List of nodes to not visit (if applicable)
        self.do_not_visit = [False for _ in range(len(G.nodes))]
        
        # Has Q-table been generated (if applicable)
        self.q_table_generated = False
        
        
    def step(self, action):
        # Take a step in the graph with  
        assert action <= self.num_nodes
        
        if (action, self.state) in self.G.edges:
        
            sampled_mean = self.mean[action]
            sampled_stddev = self.stddev[action]

            reward = np.random.normal(loc=sampled_mean, scale=sampled_stddev)
            self.state = action
            
            done = False
            
            self.G.nodes[action]['r_hist'].append(reward)
            
            # Determine what R (reward) to use in Q-learning algorithm.
            # Sampled reward or full estimate.
            if self.belief_update=='Bayesian' or self.belief_update=='average' or self.belief_update=='Bayesian_full_update'\
            or self.belief_update=='average_full_update' or self.Q_table_version=='Thompson' or self.Q_table_version=='UCB'\
            or self.local_sampling is not None:
                self.G.nodes[action]['est'].update(reward)
                QL_reward = self.G.nodes[action]['est'].get_param()[0]
            elif self.belief_update is None:
                QL_reward = reward
            else:
                raise ValueError('Invalid update method. (Must be Bayesian, average, or None.)')
            self.G.nodes[action]['n_visits'] += 1
        else:
            print(self.state)
            print(action)
            QL_reward = -100
            done = True
            print("Something is wrong. Illegal action")
            
        self.visitedStates.append(self.state)
        observation = self.state # (which is now action)
        return observation, QL_reward, done
    
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
        action = None
        N_explorations = -np.inf
        for neighbor in self.G.neighbors(node):
            if self.visits[neighbor] > N_explorations:
                N_explorations = self.visits[neighbor]
                action = neighbor
        return neighbor
    
    def visit_all_nodes(self, node):
        """
        Visits all nodes in graph (at least) once.
        
        param node: current state.
        """            
        
        # Explore neighboring nodes
        for neighbor in list(self.G.neighbors(node)):
            if neighbor != node:
                if self.visits[neighbor] == 0:
                    return neighbor
                    
        # All neighboring nodes explored
        neighbors = list(self.G.neighbors(node))
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
            self.K = round(self.N / len(self.G.nodes)) # Number of times to explore each node
            self.do_not_visit = [False for _ in range(len(self.G.nodes))]
            
        # If exploration is done (take greedy action)
        if np.sum(self.visits) >= self.N_actual:
            return np.argmax(self.q_table[node])
        
        # Not done exploring current node not 
        if self.visits[node] < self.K:
            return node
        
        # Explore neighboring nodes
        for neighbor in list(self.G.neighbors(node)):
            if neighbor != node:
                if self.visits[neighbor] < self.K:
                    return neighbor
                    
        # All neighboring nodes explored
        for neighbor in list(self.G.neighbors(node)):
            if neighbor != node:
                if not self.do_not_visit[neighbor]:
                    self.do_not_visit[neighbor] = True
                    self.N = self.N-1
                    self.K = round(self.N / len(self.G.nodes))
                    return neighbor
        print('failed')
    
    def QL_alg1_action(self, node):
        """
        TODO: Probably the same as function explore()
        
        epsilon-greedy algorithm where 'epsilon-step' chooses least visited node
        """
        action = node
        min_exploration = self.visits[node]
        for neighbor in self.G.neighbors(node):
            if self.visits[neighbor] < min_exploration:
                action = neighbor
                min_exploration = self.visits[neighbor]
        return action
    
    def local_thompson_sampling(self, node):
        """
        Local Thompson sampling
        """
        # Deciding next s using Thompson Sampling.
        muhats = []
        zs = []
        # Sample muhat from the posterior of NormalBayesianEstimation, for all s in the neighborhood(including curr_s).
        for z in list(self.G[node])+[node]:
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            muhats.append(np.random.randn()*np.sqrt(var_1)+mu_1)
            zs.append(z)
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def local_greedy(self, node):
        """
        Local greedy action
        """
        # Deciding next s using Thompson Sampling.
        muhats = []
        zs = []
        # Sample muhat from the posterior of NormalBayesianEstimation, for all s in the neighborhood(including curr_s).
        for z in list(self.G[node])+[node]:
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            muhats.append(mu_1)
            zs.append(z)
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def local_UCB(self, node):
        """
        Local UCB
        """
        beta=1 # Hard code
        muhats = []
        zs = []
        for z in list(self.G[node])+[node]:
            
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            t = self.visits[z] + 1
            h = np.sum(self.visits)
            muhats.append(mu_1+self.uncertainty * np.sqrt(np.log(h+1)/t))
            zs.append(z)
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def QL_alg4_action(self, node):
        """
        Local UCB with elimination
        """
        muhats = []
        zs = []
        for z in list(self.G[node])+[node]:
            
            if self.q_table[node, z] > -np.inf: # Do not consider eliminated state-action pairs
                
                mu_1,var_1 = self.G.nodes[z]['est'].get_param()

                muhats.append(mu_1+self.uncertainty * np.sqrt(var_1))
                zs.append(z)
            else:
                muhats.append(-np.inf)
                zs.append(z)
                
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def global_thompson_sampling(self):
        """
        Samples from posterior distributions at each node in the whole graph.
        """
        samples = []
        for node in self.G.nodes:
            mu, var = self.G.nodes[node]['est'].get_param()
            sample = np.random.randn()*np.sqrt(var)+mu
            samples.append(sample)
        return samples
    
    def get_bayesian_estimates(self):
        mus = []
        sigs = []
        for node in self.G.nodes:
            mu, var = self.G.nodes[node]['est'].get_param()
            mus.append(mu)
            sigs.append(sigs)
        return mus, sigs
        
        
    def train_agent(self, episodes = 1, H=100, init_node= None, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_discount=1,
                    T_eps_max=np.inf, QL_type=0, update_multiple_qs=False, efficient_explore_length=None,\
                   start_with_exploration=False, update_frequency=10):
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
        K = episodes
        

        # For plotting metrics
        all_states = []
        all_rewards = []

        for k in range(1, episodes+1):
            state = self.reset(init_node)

            episodes, reward = 0, 0
            done = False
            for h in range(0, H):
                all_states.append(state)


#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Determine action--------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
                if QL_type ==0: # Epsilon greedy
                    if random.uniform(0, 1) < epsilon_discount**h*epsilon and h<T_eps_max:
                            action = random.choice(list(self.G.neighbors(state)))  
                    else:
                        action = np.argmax(self.q_table[state]) # Exploit learned values
                elif QL_type == 1: # 'Greedy efficient' exploration
                    if random.uniform(0, 1) < epsilon_discount**h*epsilon and h<T_eps_max:
                        action = self.QL_alg1_action(state)
                    else:
                        action = np.argmax(self.q_table[state]) # Exploit learned values
                elif QL_type == 2:
                    if random.uniform(0, 1) < epsilon_discount**h*epsilon and h<T_eps_max:
                        action = self.local_thompson_sampling(state)
                    else:
                        action = np.argmax(self.q_table[state]) # Exploit learned values
                elif QL_type == 3:
                    if random.uniform(0, 1) < epsilon_discount**h*epsilon and h<T_eps_max:
                        action = self.local_UCB(state)
                    else:
                        action = np.argmax(self.q_table[state]) # Exploit learned values
                elif QL_type == 4:
                        if random.uniform(0, 1) < epsilon_discount**h*epsilon and h<T_eps_max:
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
                        means = np.array([self.G.nodes[i]['est'].get_param()[0] for i in self.G.nodes])
                        self.q_table = get_Q_table(self.G, means, T=rounds-efficient_explore_length)
                    action = self.explore_efficiently(state, N=N) 
                elif QL_type == 'Thompson':  
                    if h == 100:
                        # action = self.explore_efficiently(state, N=20) 
                        means = np.array([self.G.nodes[i]['est'].get_param()[0] for i in self.G.nodes])
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
                    assert (self.Q_table_version is not None) or (self.local_sampling is not None)
                    

                    if self.Q_table_version is not None:
                        if np.min(self.visits) == 0 and start_with_exploration:
                            action = self.visit_all_nodes(state)

                        else:
                            if (h%update_frequency==0):                 
                                if self.Q_table_version == 'UCB':
                                    b_ts = [i for i in range(self.num_nodes)]
                                    for node in range(self.num_nodes):
                                        t = self.visits[node] + 1
                                        b_ts[node] = self.uncertainty*np.sqrt(np.log(1+h)/t)
                                        self.b_ts = b_ts
                                    # means = [np.mean(self.collected_rewards[i]) for i in range(self.num_nodes)]
                                    means = self.get_bayesian_estimates()[0]

                                    self.means = means
                                    means = [means[i] + b_ts[i] for i in range(self.num_nodes)]

                                elif self.Q_table_version=='Thompson':
                                    means = self.global_thompson_sampling()

                                # Generate "known rewards" Q-table with mean estimates
                                self.q_table,_ , _ = get_Q_table(self.G, means, self.num_nodes)
                            action = np.argmax(self.q_table[state])
                          
                    elif self.local_sampling is not None:
                        if self.local_sampling == 'local_Thompson':
                            action = self.local_thompson_sampling(state)
                        elif self.local_sampling == 'local_UCB':
                            action = self.local_UCB(state)
                        elif self.local_sampling == 'local_greedy':
                            action = self.local_greedy(state)

#-------------------------------------------------------------------------------------#                      
#-------------------------------------------------------------------------------------#
#-------------------------------End of determine action-------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################


                # Next state, reward, and store reward
                next_state, QL_reward, done = self.step(action) 
                all_rewards.append(reward)
                self.collected_rewards[action].append(QL_reward)
                

#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Update Q-table (if applicable)------------------------#
#---------NOTE: This section is skipped for Q-table + UCB/Thompson approach-----------#
#-------------------------------------------------------------------------------------#

                if not (QL_type is None):
                    old_value = self.q_table[state, action]
                    next_max = np.max(self.q_table[next_state])

                    if self.belief_update!='Bayesian_full_update' and self.belief_update!='average_full_update':
                        new_value = (1 - alpha) * old_value + alpha * (QL_reward + gamma * next_max)

                    else:
                        new_value = self.G.nodes[action]['est'].get_param()[0] + gamma * next_max # Full Bayesian update

                    # Do not update if using 'finite-time' QL graph theory, or QL_UCB
                    if (QL_type != 8) and (QL_type != 'Thompson'):
                        self.q_table[state, action] = new_value

                    # Update all state-'action' pairs with 'action'=action
                    if (QL_type != 8) and (QL_type != 'Thompson'):
                        if update_multiple_qs:
                            for node in self.G.neighbors(action):
                                old_value = self.q_table[node, action]
                                if self.belief_update!='Bayesian_full_update' and self.belief_update!='average_full_update':
                                    new_value = (1 - alpha) * old_value + alpha * (QL_reward + gamma * next_max)
                                else: 
                                    new_value = self.G.nodes[action]['est'].get_param()[0] + gamma * next_max # Full Bayesian update
                                self.q_table[node, action] = new_value
                                self.G.nodes[node]['actions'][(node, next_state)] += 1 # Update state-action count



                    mu_temp, sigma_temp = self.G.nodes[action]['est'].get_param()
                    # Update upper and lower confidence bounds
                    self.upper95[action] = mu_temp + self.uncertainty*sigma_temp
                    self.lower95[action] = mu_temp - self.uncertainty*sigma_temp
                    if  self.upper95[action] < self.lower95[action]:
                        print('Something is wrong. UCB<LCB')

                    if QL_type == 4: # Eliminate high-confidence "bad" choice
                        lower_confidence = list(self.lower95.values())
                        lower_confidence = np.array(lower_confidence)
                        if (self.upper95[action] < lower_confidence[list(self.G.neighbors(action))]).any():
                            self.q_table[state, action] = -np.inf
                            
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------End of update Q-table---------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################



#######################################################################################
#-------------------------------------------------------------------------------------#
#-------------------------------Update parameters-------------------------------------#

                self.G.nodes[state]['actions'][(state, next_state)] += 1 # Update state-action count
                state = next_state
                self.G.nodes[state]['n_visits'] += 1
                self.visits[state] += 1
                episodes += 1
                
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#######################################################################################

                
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
        
        return mu_max - mu_visited
            
        
        
    
        
