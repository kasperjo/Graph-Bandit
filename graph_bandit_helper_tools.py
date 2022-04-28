import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import graph_bandit_RL
from tqdm import trange

def end_node_success_rate(regrets):
    """
    param regrets: N x M array, where N are number of samples and M are number of time steps, for N samples of computing the 
    regret over M time steps.
    
    returns: the fraction of the N samples, out of which the agent ended up at the best node
    
    NOTE: regrets can also be formatted as a dictionary with the N x M regret format for various 'approaches'
    """
    try:
        for key in regrets.keys():
            print(key + ': ' +  str(np.sum(regrets[key][:,-1:] == 0)/regrets[key].shape[0]))
    except:
        n_regrets = regrets.shape[0]
        total_failed = 0
        for i in range(regrets.shape[0]):
            if regrets[i,-1] != 0:
                total_failed += 1
        return (n_regrets-total_failed)/n_regrets
def found_best_node_success_rate(environments):
    n_samples = len(environments)
    n_success = 0
    
    for env in environments:
        G = env.G.copy()
        best_node_true = np.argmax(env.mean)
    
        best_node = None
        try: # Check Bayesian estimates
            mu_best = -np.inf
            for node in G.nodes:
                mu_est, var_est = G.nodes[node]['est'].get_param()
                if mu_est > mu_best:
                    best_node = node
                    mu_best = mu_est
        except: # Try Q-table (for Q-learning methods)
            Q_table = env.q_table
            Q_best = -np.inf
            for node in range(Q_table.shape[0]):
                Q_val = Q_table[node,node]
                if Q_val > Q_best:
                    best_node = node
                    Q_best = Q_val
        if best_node == best_node_true:
            n_success += 1
    return n_success / n_samples
        
        

def return_graph(graph_type='fully_connected', n_nodes=6, n_children=None):
    """
    Returns specified graph type.
    
    param graph_type: string. fully_connected, line, circle, star, or tree
    param n_nodes: Number of nodes in graph.
    param n_children: Number of children per node in the tree graph (only applicable for tree graph)
    """
    G = nx.Graph()

    if graph_type=='fully_connected':
        for i in range(n_nodes):
            for j in range(n_nodes):
                G.add_edge(i,j)
    elif graph_type=='line' or graph_type=='circle':
        for i in range(n_nodes):
            G.add_edge(i,i)
            if i<n_nodes-1:
                G.add_edge(i,i+1)
        if graph_type=='circle':
            G.add_edge(0,n_nodes-1)
    elif graph_type=='star':
        G.add_edge(0,0)
        for i in range(1,n_nodes):
            G.add_edge(i,i)
            G.add_edge(0,i)
    elif graph_type=='tree':
        assert n_children is not None
        G.add_edge(0,0)
        children = {0:0}
        for i in range(1,20):
            G.add_edge(i,i)
            available_nodes = np.sort(list(G.nodes))
            for node in available_nodes:
                if children[node] < n_children:
                    G.add_edge(node,i)
                    children[node] += 1
                    children[i] = 0
                    break
    else:
        raise ValueError("Invalid graph type. Must be fully_connected, line, circle, star, or tree.")
    return G
        

def draw_graph(G, zero_indexed=True):
    """
    Draws graph.
    
    param G: networkx graph.
    param zero_indexed: if True, nodes are zero-indexed, else indexing starts at one.
    """
    if zero_indexed:
        labels = {n:n for n in G.nodes}
    else:
        labels = {n:n+1 for n in G.nodes}
    nx.draw(G,labels=labels)
    plt.show()
    
    

    
    
def plotRegrets(allRegrets, titles=None):
    """
    param allRegrets: list of dictionaries of regrets(samples, time_steps) for various algorithms and runs
    """
    nPlots = len(allRegrets)
    sns.set()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'grey', 'olive', 'cyan']
    
    labels = {'greedy': 'Local greedy', 'thompson': 'Local Thompson sampling', 'UCB': 'Local UCB','Q_learning': 'Q-learning',\
              'Q_table_UCB': 'Q-table (global) UCB', 'Q_table_Thompson': 'Q-table (global) Thompson sampling'
             }
    
    if nPlots == 1:
        regrets = allRegrets[0]
        for i, key in enumerate(regrets.keys()):
            regret = regrets[key]
            regret = np.cumsum(regret, axis=1)
            df = pd.DataFrame(np.transpose(regret))
            df.columns = [labels[key] for _ in range(regret.shape[0])]
            sns.lineplot(data=df, ci=None, palette=[colors[i]])
        if titles is not None:    
            plt.title(titles[0])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(17,10))
        for j, regrets in enumerate(allRegrets):
    #         plt.subplot(1, nPlots, j+1)
    #         plt.figure(figsize=(5, 5))
            for i, key in enumerate(regrets.keys()):
                regret = regrets[key]
                regret = np.cumsum(regret, axis=1)
                df = pd.DataFrame(np.transpose(regret))
                df.columns = [labels[key] for _ in range(regret.shape[0])]
                if j<3:
                    sns.lineplot(ax = axes[0,j], data=df, ci=None, palette=[colors[i]])
                    axes[0,j].set_title(titles[j])
                else:
                    sns.lineplot(ax = axes[1,j%3], data=df, ci=None, palette=[colors[i]])
                    axes[1,j%3].set_title(titles[j])
            #plt.title(titles[j])
        #plt.legend()
    plt.show()
    

def testLearning(episodes, T, n_samples, epsilon, epsilon_discount, algorithms, G,\
                 means=None, stdevs=None, mean_magnitude=1, stdev_magnitude=1,\
                 init_nodes=None, QL_type=0, update_multiple_qs=False,\
                 efficient_explore_length=None, prior_uncertainty=10):
    """
    param episodes: number of episodes
    param T: time horizon (per epoch)
    param n_samples: number of randomized samples
    param epsilon: exploration parameter (epsilon greedy)
    param epsilon: epsilon discount; epsilon=v^t * epsilon0
    param algorithms: list of "exploration algorithms" to evaluate; rlNoUpdate, rlBayesianUpdate, rlAverageUpdate,
        rlBayesianFull, (rlAverageFull), thompson, UCB
    param G: networkx graph
    param T_eps_max: maximum exploration length
    param mean_magnitude: magnitude of mean reward; e.g. mean_magnitude=a-> means=np.random.normal(size=(n_samples,6))*a
    param stdev_magnitude: magnitude of mean reward; e.g. stdev_magnitude=a-> means=np.ones((n_samples,6))*10*a
    param inidNodes: Dictionary of initial nodes for each algorithm
    """
    all_env = {alg: [] for alg in algorithms}
    
    nNodes = len(G.nodes)
    regrets = {alg: np.zeros((n_samples, T)) for alg in algorithms}
    if means is None:
        means = np.random.normal(size=(n_samples,nNodes))*mean_magnitude
    if stdevs is None:
        stdevs = np.ones((n_samples,nNodes))*stdev_magnitude
        
    if init_nodes is None:
        init_nodes = {alg: None for alg in algorithms}

    for i in trange(n_samples): 
        
        # Local Thompson sampling
        if 'thompson' in algorithms:
            Thompson = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update=None,\
                                                        bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                        stdev_magnitude**2], local_sampling='local_Thompson')
            Thompson.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                       QL_type=None, init_node=init_nodes['thompson'])
            regrets['thompson'][i,:] = Thompson.expectedRegret()
            all_env['thompson'].append(Thompson)

            
        # Local UCB
        if 'UCB' in algorithms:
            UCB = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update=None,\
                                                        bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                        stdev_magnitude**2], local_sampling='local_UCB')
            UCB.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                       QL_type=None, init_node=init_nodes['UCB'])
            regrets['UCB'][i,:] = UCB.expectedRegret()
            all_env['UCB'].append(UCB)

            
        # Local Greedy action
        if 'greedy' in algorithms:
            greedy = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update=None,\
                                                        bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                        stdev_magnitude**2], local_sampling='local_greedy')
            greedy.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                       QL_type=None, init_node=init_nodes['greedy'])
            regrets['greedy'][i,:] = greedy.expectedRegret()
            all_env['greedy'].append(greedy)

            
         
        # Q-learning
        if 'Q_learning' in algorithms:
            QL = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update='Bayesian_full_update',\
                                                        bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                        stdev_magnitude**2])
            QL.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                       QL_type=1, init_node=init_nodes['Q_learning'],\
                                       update_multiple_qs=True)
            regrets['Q_learning'][i,:] = QL.expectedRegret()
            all_env['Q_learning'].append(QL)
         
        # Q-table approach with UCB estimates
        if 'Q_table_UCB' in algorithms:
                Q_table_UCB = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update=None,\
                                                            bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                            stdev_magnitude**2], Q_table_version='UCB')
                Q_table_UCB.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                           QL_type=None, init_node=init_nodes['Q_table_UCB'],\
                                        update_multiple_qs=True, efficient_explore_length=efficient_explore_length)
                regrets['Q_table_UCB'][i,:] = Q_table_UCB.expectedRegret()
                all_env['Q_table_UCB'].append(Q_table_UCB)
                
        # Q-table approach with Thompson sampling
        if 'Q_table_Thompson' in algorithms:
                Q_table_Thompson = graph_bandit_RL.GraphBandit(means[i], stdevs[i], G, belief_update=None,\
                                                            bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                            stdev_magnitude**2],\
                                                               Q_table_version='Thompson')
                Q_table_Thompson.train_agent(episodes=episodes, H=T, epsilon=epsilon, epsilon_discount=epsilon_discount,\
                                           QL_type=None,\
                                             init_node=init_nodes['Q_table_Thompson'],\
                                           update_multiple_qs=True, efficient_explore_length=efficient_explore_length)
                regrets['Q_table_Thompson'][i,:] = Q_table_Thompson.expectedRegret()
                all_env['Q_table_Thompson'].append(Q_table_Thompson)
        
    return regrets, all_env

