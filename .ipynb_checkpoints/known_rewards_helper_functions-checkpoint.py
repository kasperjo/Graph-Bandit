import numpy as np
import networkx as nx

def get_Q_table(G, means, T=100):
    """
    param G: networkx graph (undirected); nodes in G = 0,1,2,3,...
    param means: vector of mean rewards; corresponding, by indexing, to graphs in G
    param T: number of times the agent plays the graph bandit "game"
    
    returns: Q-table, k, all_calls (the total number of q_value calculations)
    
    The value function is sum of all rewards over the T time steps (starting at initial node)
    """
    # Find best node and initialize Q-table
       
    n_nodes = len(means)
    best_node = np.argmax(means)
    mu_b = means[best_node]
    Q = np.ones((n_nodes,n_nodes))*(-np.inf)
    Q[best_node,best_node] = T*mu_b
    
    k=0
    next_round = {best_node}
    n_calls = 0
    while next_round:
        if k > T:
            break
            
        curr_round = next_round.copy()
        next_round = set()
        for curr_node in curr_round:
            for node in G.neighbors(curr_node):
                n_calls += 1
                q_value =  np.max(Q[curr_node])- mu_b + means[node]
                if q_value > np.max(Q[node]):
                    next_round.add(node)
                Q[node, curr_node] = q_value
        k+=1
    
    return Q, k, n_calls

    
    
        
    
        
        
    
