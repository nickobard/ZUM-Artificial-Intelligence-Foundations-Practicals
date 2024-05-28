import numpy as np
from utils import RIGHT, UP, LEFT, DOWN


# ----------------------------------------------------------------------------------------------------------------------
# Base MDP

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [Page 646]"""

    def __init__(self, actions_list,
                 terminals, transitions, reward, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.actions_list = actions_list
        self.terminals = terminals
        self.transitions = transitions
        self.reward = reward
        self.gamma = gamma

    def R(self, state):
        """Return a numeric reward for this state."""
        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""
        return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actions_list


# ----------------------------------------------------------------------------------------------------------------------
# Grid MDP

class GridMDP(MDP):

    def __init__(self, start_state, grid, action_distribution, terminals, gamma=.9):
        # grid.reverse()  # because we want row 0 on bottom, not on top
        reward = dict()
        self.states = set()
        # states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        y = 0
        for x in range(self.cols):
            for z in range(self.rows):
                self.states.add((x, y, z))
                reward[(x, y, z)] = grid[z][x]

        actions_list = [LEFT, UP, DOWN, RIGHT]
        self.action_distribution = action_distribution
        terminals = [(col, 0, row) for (row, col) in terminals]

        transitions = dict()
        for s in self.states:
            transitions[s] = dict()
            for a in actions_list:
                transitions[s][a] = self.calculate_T(s, a)
        super().__init__(actions_list=actions_list,
                         terminals=terminals, transitions=transitions,
                         reward=reward, gamma=gamma)

    def calculate_T(self, state, action):
        return [(prob, self.go(state, transform(action))) for prob, transform in self.action_distribution]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""
        if isinstance(direction, tuple):
            direction = np.array(direction)
        if isinstance(state, tuple):
            state = np.array(state)
        go_state = tuple(state + direction)
        return go_state if go_state in self.states else tuple(state)


# ----------------------------------------------------------------------------------------------------------------------
# Value iteration

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. [Equation 17.4]"""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""

    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))


def value_iteration(mdp, epsilon=1e-3):
    utilities_history = []
    policies_history = []

    U_current = mdp.reward.copy()
    utilities_history.append(U_current.copy())
    policies_history.append(best_policy(mdp, U_current))

    actions, R, T, gamma = mdp.actions, mdp.R, mdp.T, mdp.gamma
    while True:
        delta = 0.0
        U_previous = U_current.copy()
        for s in mdp.states:
            U_current[s] = R(s) + gamma * max(sum(p * U_previous[s1] for (p, s1) in T(s, a))
                                              for a in mdp.actions(s))
            delta = max(delta, abs(U_current[s] - U_previous[s]))
        utilities_history.append(U_current.copy())
        policies_history.append(best_policy(mdp, U_current))
        if delta <= epsilon * (1 - gamma) / gamma:
            return utilities_history, policies_history


# ----------------------------------------------------------------------------------------------------------------------
# Policy iteration
# TODO
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass
