import pprint

import numpy as np
from utils import RIGHT, UP, LEFT, DOWN
from utils import get_grid, grid_1, get_action_distribution, add_tuple_vectors
import random


# ----------------------------------------------------------------------------------------------------------------------
# Base MDP

class MDP:
    """A Markov Decision Process, defined by a transition model, set of all states, an action list, rewards, terminals
    (obstacle and finish states), and gamma value used later in algorithms. The transition model is
    represented by T(state, action) to return a list of (probability of applied action, state after transition)
    pairs."""

    def __init__(self, actions_list,
                 terminals, transitions, rewards, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.actions_list = actions_list
        self.terminals = terminals
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma

    def R(self, state):
        """Return a numeric reward for this state."""
        return self.rewards[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""
        return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states."""

        if state in self.terminals:
            return [None]
        else:
            return self.actions_list


# ----------------------------------------------------------------------------------------------------------------------
# Grid MDP

class GridMDP(MDP):
    """
    A Markov Decision Process for the 2D grid environment. Agent in this environment can go into four directions -
    left, up, down and right. For every action there is also defined an action distribution, which tells the
    probability of going to the intended direction.
    """

    def __init__(self, grid, action_distribution, terminals, gamma=.9):
        rewards = dict()
        self.states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])

        # convert grid into states with associated reward values.
        for x in range(self.cols):
            for y in range(self.rows):
                self.states.add((x, y))
                rewards[(x, y)] = grid[y][x]

        # list of all actions that may be executed from each non-terminal state.
        actions_list = [LEFT, UP, DOWN, RIGHT]
        self.action_distribution = action_distribution

        transitions = dict()
        # create transition model
        for s in self.states:
            transitions[s] = dict()
            for a in actions_list:
                transitions[s][a] = self.calculate_T(s, a)

        super().__init__(actions_list=actions_list,
                         terminals=terminals, transitions=transitions,
                         rewards=rewards, gamma=gamma)

    def calculate_T(self, state, action):
        """Function to calculate a transition model for the 2D environment."""
        return [(prob, self.go(state, transform(action))) for prob, transform in self.action_distribution]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""
        go_state = add_tuple_vectors(state, direction)
        return go_state if go_state in self.states else tuple(state)


# ----------------------------------------------------------------------------------------------------------------------
# Value iteration

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action."""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing action a in state s, according to the MDP and U."""
    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))


def value_iteration(mdp, **params):
    """
    Value iteration algorithm for MDP. It takes an MPD and epsilon parameters, and iterates utilities and policies
    until the utility converges enough. Utilities and policy histories are stored for interactive visualizations.
    There is a maximum limit for iterations, so it has some execution time limit.
    """
    # iterations limit
    iter_limit = 1000

    utilities_history = []
    policies_history = []

    # store the default utility and policy
    U_current = mdp.rewards.copy()
    utilities_history.append(U_current.copy())
    policies_history.append(best_policy(mdp, U_current))

    actions, R, T, gamma = mdp.actions, mdp.R, mdp.T, mdp.gamma

    i = 0
    while i < iter_limit:
        delta = 0.0
        U_previous = U_current.copy()
        for s in mdp.states:
            # compute utility for the current state
            U_current[s] = R(s) + gamma * max(sum(p * U_previous[s1] for (p, s1) in T(s, a))
                                              for a in mdp.actions(s))
            # update delta if absolute difference of two last utilities is the highest
            delta = max(delta, abs(U_current[s] - U_previous[s]))

        # store current utility and policy
        utilities_history.append(U_current.copy())
        policies_history.append(best_policy(mdp, U_current))

        # check if utilities converged enough
        if delta <= params['epsilon'] * (1 - gamma) / gamma:
            return utilities_history, policies_history
        i += 1


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    grid_data = get_grid(-1.0, 1.0, -0.04, grid_structure_fn=grid_1)
    mdp = GridMDP(grid_data['grid'], get_action_distribution(0.8), terminals=grid_data['terminals'],
                  gamma=0.9)
    print("Terminals:", mdp.terminals)
    print("Actions list:", mdp.actions_list)
    print("Transitions:", mdp.transitions)
    print("Rewards:", mdp.rewards)
