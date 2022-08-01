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

def all_paths(G, start_node, end_node):
    """
    returns a list of all paths starting at start_node and ending at end_node
    """
    def depth_first_search(u, v):
        if u in visited:
            return
        visited.append(u)
        curr_path.append(u)
        if u==v:
            all_paths.append(curr_path.copy())
            visited.remove(u)
            curr_path.pop()
            return
        neighbors = G.neighbors(u)
        try:
            neighbors.remove(u)
        except:
            None
        for node in neighbors:
            depth_first_search(node, v)
        curr_path.pop()
        visited.remove(u)
    
    visited = []
    curr_path = []
    all_paths = []
    depth_first_search(start_node, end_node)
    return all_paths

    
        

def brute_force_policies(G, means, T=100):
    """
    param G: networkx graph (undirected); nodes in G = 0,1,2,3,...
    param means: vector of mean rewards; corresponding, by indexing, to graphs in G
    param T: number of times the agent plays the graph bandit "game"
    
    returns: dictionary {node: [best_policy, total_reward]} for each node in G
    conditioned on ending up at the best node; best_policy is a list of nodes, staring at 
    node and ending at the best node in G
    
    Note: Enforces each path to end up at the best node in G;
    i.e. T must be large enough, so that the best node can be reached from each other node in G
    """
    # Find best node and initialize Q-table
    means = np.reshape(means, (-1,))
    n_nodes = means.shape[0]
    best_node = np.argmax(means)
    mu_b = means[best_node]
    
    optimal_policies = {}
    for node in G.nodes:
        all_paths_temp = all_paths(G, node, best_node)
        best_path = None
        best_reward = -np.inf
        for path in all_paths_temp:
            if len(path) > T:
                pass
            else:
                n_steps_at_mu_b = T - len(path)
                path = path + [best_node for _ in range(n_steps_at_mu_b)]
                reward_temp = np.sum(means[path])
                if reward_temp > best_reward:
                    best_path = path.copy()
                    best_reward = reward_temp
        
        if best_path is None:
            optimal_policies[node] = [np.nan, -np.inf]
        else:
            optimal_policies[node] = [best_path, np.sum(means[best_path])]
    
    return optimal_policies

def get_paths(Q_table, best_node, T):
    """
    param Q_table: a Q_table of state-action values
    best_node: best node in corresponding graph
    param T: number of times the agent plays the graph bandit "game"
    
    returns: dictionary {node: [path, total_reward]}
    node is a node/state; path is a path/list starting at node
    and ending at the best node in the graph G, corresonding to the Q_table (see function compute_Q_table);
    total_reward is the accumulated expected reward along path
    """
    
    n_nodes = Q_table.shape[0]
    
    paths = {}
    for node in range(n_nodes):
        if np.max(Q_table[node]) == -np.inf:
            paths[node] = [np.nan, -np.inf]
        else:
            path = [node]
            curr_node = node
            while curr_node != best_node:
                curr_node = np.argmax(Q_table[curr_node])
                path.append(curr_node)
            n_steps_at_mu_b = T - len(path)
            path = path + [best_node for _ in range(n_steps_at_mu_b)]
            paths[node] = [path, np.max(Q_table[node])]
        
    return paths


def control_functions(Gs, means, T):
    """
    Checks if brute force method yields same result as the more sofisticated_method
    
    param Gs: list of networkx graphs to evaluate or networkx graph
    param means: list of means vector or means vector
    param T: number of times the agent plays the graph bandit "game"
    
    returns: True if succesfull; False otherwise.
    """
    if type(Gs) is not list:
        paths_brute_force = brute_force_policies(Gs, means, T)
        Q = get_Q_table(Gs, means, T)
        paths_sofisticated = get_paths(Q, np.argmax(means), T)
        if len(paths_brute_force) != len(paths_sofisticated):
            return False
        for i in range(len(paths_brute_force)):
            if paths_brute_force[i] != paths_sofisticated[i]:
                 return False
    else:
        for i, G in enumerate(Gs):
            paths_brute_force = brute_force_policies(G, means[i], T)
            Q = get_Q_table(G, means[i], T)
            paths_sofisticated = get_paths(Q, np.argmax(means[i]), T)
            if len(paths_brute_force) != len(paths_sofisticated):
                return False
            for i in range(len(paths_brute_force)):
                if paths_brute_force[i] != paths_sofisticated[i]:
                     return False
    
    return True
    
    
        
    
        
        
    
