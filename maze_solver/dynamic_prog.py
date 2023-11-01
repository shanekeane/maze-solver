#Functions for solving a maze with dynamic programming.
#Methodology outlined in Chapter 4 of Sutton and Barto.

import numpy as np
from copy import deepcopy

def get_possible_actions_for_state(s, grid_size):
    """
    Gets all possible states after moving up/down/left/right
    
    Parameters
    ----------
    s : int
        The state number.
    grid_size : int
        The side size of square grid. 
 
    Returns
    -------
    numpy.ndarray
        An array of ints describing all possible states after a move 
        (up,down,left,right)
    """
    up = s-grid_size
    down = s+grid_size
    left = s-1
    right = s+1
    if up<0:
        up = s
    if down>=grid_size**2:
        down = s
    if s%grid_size==0:
        left = s
    if s%grid_size==grid_size-1:
        right = s
        
    actions = np.asarray([up, down, left, right])
    return actions
    
def get_pi_for_state(state, V, rewards, GAMMA, choose_one=False):
    """
    Calculates policy for individual state. Total states S=len(V)
    because the last state is the goal and is excluded. 
    
    Parameters
    ----------
    state : int
        The state number.
    V : numpy.ndarray, shape = (S)
        An array of floats containing the state-value functions.
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    choose_one: bool
        Determines whether to choose a single next state for each state, or
        leave a choice based on probability for each possible next state.
 
    Returns
    -------
    pi_state : numpy.ndarray, shape = (S)
        Returns an array indicating probability for moving to each of the S
        states.
    """
    rewards = rewards.flatten()
    pi_state = np.zeros(len(V))
    grid_size = int(np.sqrt(len(V)))
    possible_actions = get_possible_actions_for_state(state, grid_size)
    vs_for_as = V[possible_actions]
    rs_for_as = rewards[possible_actions]
    sum_values = rs_for_as + GAMMA*vs_for_as
    if choose_one==False:
        max_as = np.where(sum_values==np.max(sum_values))[0]
        best_as = possible_actions[max_as]
        for best_a in best_as:
            pi_state[best_a] += 1.0/len(best_as)
    else:
        best_a = possible_actions[np.argmax(sum_values)]
        pi_state[best_a] = 1.0
    return pi_state

def get_policy_from_V(V, rewards, GAMMA, choose_one=False):
    """
    Returns policy pi from an inputted array of state-value functions.
    There are S states. 
    
    Parameters
    ----------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the state-value functions.
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    choose_one: bool
        Determines whether to choose a single next state for each state, or
        leave a choice based on probability for each possible next state.
 
    Returns
    -------
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containgan array 
        indicating probability for moving to each of the S states. 
    """
    rewards = rewards.flatten()
    pi = np.zeros((len(V), len(V)))
    for state in range(0, len(V)-1):
        pi_for_state = get_pi_for_state(state, V, rewards, GAMMA, choose_one)
        pi[state] = pi_for_state
    return pi

def policy_evaluation(V, pi, rewards, GAMMA, DELTA):
    """
    Returns policy pi from an inputted array of state-value functions.
    There are S states. 
    
    Parameters
    ----------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the state-value functions.
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containgan array 
        indicating probability for moving to each of the S states.
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    DELTA : float
        Sets maximum difference between old and new Vs during policy evaluation.
 
    Returns
    -------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the updated state-value functions.
    """
    rewards = rewards.flatten()
    diff = DELTA+1.0
    while diff>DELTA:
        for i in range(0, len(V)-1):
            v = deepcopy(V)
            V[i] = np.sum(pi[i]*(rewards+GAMMA*v))
        diff = np.linalg.norm(v-V)
    return V

def policy_iteration(rewards, GAMMA, DELTA):
    """
    Does policy iteration on grid defined by rewards.
    
    Parameters
    ----------
    rewards : numpy.ndarray, shape = (S) or shape = (sqrt(S), sqrt(S))
        An array of floats containing the rewards for transitions to each state.
    GAMMA : float
        The discount rate.
    DELTA : float
        Sets maximum difference between old and new Vs during policy evaluation.
 
    Returns
    -------
    V : numpy.ndarray, shape = (S)
        An array of floats containing the calculated state-value functions.
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containgan array 
        indicating probability for moving to each of the S states.     
    """
    rewards = rewards.flatten()
    V = np.zeros(len(rewards))
    pi = get_policy_from_V(V, rewards, GAMMA, choose_one=False)
    old_pi = np.ones_like(pi)
    while (old_pi==pi).all()==False:
        old_pi = deepcopy(pi)
        V = policy_evaluation(V, pi, rewards, GAMMA, DELTA)
        pi = get_policy_from_V(V, rewards, GAMMA, choose_one=True)
    return V, pi
