#Functions for solving a maze with Q learning
#Methodology outlined in Chapter 6 of Sutton and Barto.

import numpy as np
from copy import deepcopy


def get_possible_actions(s, grid_size):
    """
    For maze and state, actions which can be made
    
    Parameters
    ----------
    s : int
        state
    grid_size : int
        Side length of maze
 
    Returns
    -------
    ndarray : int
        Array of type int, possible actions
    """
    next_states=list()
    if s>=grid_size: #up
        next_states.append(0)
    if s<(grid_size**2)-grid_size: #down
        next_states.append(1)
    if s%grid_size!=0: #left
        next_states.append(2)
    if s%grid_size!=grid_size-1:
        next_states.append(3)
        
    return np.asarray(next_states)

def get_impossible_actions(s, grid_size):
    """
    For maze and state, actions which can't be made due to walls of maze
    
    Parameters
    ----------
    s : int
        state
    grid_size : int
        Side length of maze
 
    Returns
    -------
    ndarray : int
        Array of type int, impossible actions
    """
    next_states=list()
    if s<grid_size: #up
        next_states.append(0)
    if s>=(grid_size**2)-grid_size: #down
        next_states.append(1)
    if s%grid_size==0: #left
        next_states.append(2)
    if s%grid_size==grid_size-1:
        next_states.append(3)
        
    return np.asarray(next_states, dtype=int)

def generate_q(maze):
    """
    For a maze, initialize Q array of size states x actions (4)
    
    Parameters
    ----------
    maze : ndarray
        Maze defined with paths reward ints -1, walls rewards -1000
 
    Returns
    -------
    ndarray : float
        Q values. Impossible states are given value -1e-24
    """
    grid_size = maze.shape[0]
    q = np.zeros((grid_size**2, 4))
    for i in range(grid_size**2):
        imposs = get_impossible_actions(i, grid_size)
        q[i][imposs] = -1e24
        
    return q

def take_action(state, Q, EPSILON):
    """
    Return action for state in accordance with epsilon-greedy policy.
    
    Parameters
    ----------
    state : int
        current state
    Q : ndarray
        Q array of floats, size states x actions (4)
    EPSILON : float
        Indicating amount of exploration in epsilon-greedy policy (0-1). O is zero exploration.
 
    Returns
    -------
    int
        next action 0/1/2/3 indicating up/down/left/right
    """
    
    rand = np.random.uniform()
    if rand > EPSILON:
        return np.argmax(Q[state])
    else:
        grid_size = int(np.sqrt(Q.shape[0]))
        poss_actions = get_possible_actions(state, grid_size)
        return int(np.random.choice(poss_actions))

def get_next_state(state, action, grid_size):
    """
    Take action and return next state
    
    Parameters
    ----------
    state : int
        current state
    action : int
        0/1/2/3 indicating up/down/left/right
 
    Returns
    -------
    int
        next state
    """
    if action == 0:
        return state-grid_size
    elif action == 1:
        return state+grid_size
    elif action == 2:
        return state-1
    elif action == 3:
        return state+1

def get_route(Q, maze, start_x=0, start_y=0):
    """
    For maze and associated Q values, return array of states passed with greedy policy
    
    Parameters
    ----------
    Q : ndarray
        Q values of type float, of size states x actions (4)
    maze : ndarray
        Maze defined with paths reward ints -1, walls rewards -1000
    start_x : starting x point on maze grid 
    start_y : starting y point on maze grid
 
    Returns
    -------
    ndarray : int
        Sequence of all states passed to terminal state.
    """
    grid_size=maze.shape[0]
    state=start_x+grid_size*start_y #start state
    sequence = list()
    sequence.append(state)
    while state != len(maze.flatten())-1:
        action = take_action(state, Q, 0) #greedy
        state = get_next_state(state, action, grid_size)
        sequence.append(state)
        
    return sequence

def solve(maze, GAMMA, ALPHA, EPSILON, DELTA):
    """
    For a maze defined by maze
    
    Parameters
    ----------
    maze : int
        Maze defined with paths reward -1, walls rewards -1000
    GAMMA : float
        Discount factor (0-1)
    ALPHA : float
        Alpha parameters in Q learning (0-1)
    EPSILON : float
        Parameter defining amount of exploration (0-1). 0 is no exploration
    DELTA : float
        Max diff between old and new for termination. 
 
    Returns
    -------
    ndarray : float 
        A grid with size state x actions (4) indicating Q values
    """
    r = maze.flatten() # rewards vector from maze
    r[r==-100]=-1000
    grid_size = maze.shape[0]
    Q = generate_q(maze)
    Q_old = np.ones_like(Q)
    while np.max(np.abs(Q-Q_old)) > DELTA:
        Q_old = deepcopy(Q)
        state = 0
        while state != grid_size**2-1:
            action = take_action(state, Q, EPSILON)
            next_state = get_next_state(state, action, grid_size)
            Q[state,action] += ALPHA*(r[next_state] + GAMMA*np.max(Q[next_state]) - Q[state,action])
            state = deepcopy(next_state)
    return Q
