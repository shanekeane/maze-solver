import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#from maze import Maze
from .q_learning import get_route

def generate_maze(grid_size):
    """
    Generate a square maze.
    
    Parameters
    ----------
    grid_size : int
        The side length of the square grid/maze.
 
    Returns
    -------
    big_maze : numpy.ndarray, shape = (grid_size, grid_size)
        An array of floats contains the reward values for each point
        in the grid, which defines the maze. 
    """
    if grid_size%2==0:
        raise ValueError("Choose an odd number for grid size.")
        
    np.random.seed(2)
    maze=np.zeros(shape=(grid_size, grid_size))
    start_point=np.array([0,0])
    maze_generator=Maze(maze, start_point)
    big_maze = maze_generator.maze
    #big_maze[grid_size-2,grid_size-1]=big_maze[grid_size-1,grid_size-1]=2
    big_maze[big_maze==2.] = -1
    big_maze[big_maze==0.] = -100
    plt.imshow(big_maze, cmap = 'Greys_r')
    
    return big_maze


def get_actions_from_pi(pi):
    """
    From a policy, determine where to move next from each of S states.
    
    Parameters
    ----------
    pi : numpy.ndarray, shape = (S, S)
        The policy. Each row corresponds to a state, and containgan array 
        indicating probability for moving to each of the S states.
 
    Returns
    -------
    actions : numpy.ndarray, shape = (S-1)
        An array containing the next state for each state other than the
        terminal state (bottom right).
    """
    actions = list()
    for i in range(pi.shape[1]-1):
        actions.append(np.random.choice(np.arange(pi.shape[1]), p=pi[i]))
    return np.asarray(actions)

def plot_maze_solution(pi, old_maze, start_i=0, start_j=0):
    """
    Given the actions, maze, and start points, plot the maze with its solution.
    There are S states in a maze of side length L.
    
    Parameters
    ----------
    greedy_actions : numpy.ndarray, shape = (S-1)
        The next state for each state other than the terminal state.
    old_maze : numpy.ndarray, shape = (L, L)
        The rewards for transition to each state.
    start_i : int
        Starting first index.
    start_j : int
        Starting second index.
 
    Returns
    -------
    None
    """
    grid_size = old_maze.shape[0]
    old_maze = old_maze.flatten()
    maze = deepcopy(old_maze)
    actions = get_actions_from_pi(pi)
    
    state = start_j+grid_size*start_i
    while state != len(old_maze)-1:
        maze[state] = -50
        state = actions[state]
    maze[state] = -50
    
    #Plot maze together with solution
    f, mazes = plt.subplots(1,2)
    mazes[0].imshow(old_maze.reshape(grid_size, grid_size), cmap='Greys_r')
    mazes[1].imshow(maze.reshape(grid_size, grid_size), cmap = 'Greys_r')

def plot_solution_from_q(Q, maze, start_x=0, start_y=0):
    """
    Plots original maze next to maze with solution
    
    Parameters
    ----------
    Q : ndarray
        Q values of type float, of size states x actions (4)
    maze : ndarray
        Maze defined with paths reward ints -1, walls rewards -1000
    start_x : starting x point on maze grid 
    start_y : starting y point on maze grid
    """
    route=get_route(Q, maze, start_x, start_y)
    grid_size=maze.shape[0]
    maze1=maze.flatten()
    maze1[route]=-50
    maze1=maze1.reshape(grid_size, grid_size)
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(maze, cmap='binary_r')
    axs[1].imshow(maze1, cmap='binary_r')
