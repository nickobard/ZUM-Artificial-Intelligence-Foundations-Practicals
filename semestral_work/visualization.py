from matplotlib.lines import Line2D

from mdp import GridMDP, value_iteration
from utils import get_action_distribution, get_grid_1, keep_direction
from utils import UP
import ipywidgets as widgets
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque
import numpy as np
import matplotlib.gridspec as gridspec


class GridMDPVisualizer:
    def __init__(self, get_grid_fn, iteration_algorithm_fn, *, terminal_reward_min=-10.0, terminal_reward_max=10.0):

        self.get_grid_fn = get_grid_fn
        self.iteration_algorithm_fn = iteration_algorithm_fn

        self.terminal_reward_min = terminal_reward_min
        self.terminal_reward_max = terminal_reward_max

        self.grid_data = None
        self.hparams = None
        self.mdp = None
        self.utilities = None
        self.policies = None

    def get_visualization_data(self):
        self.grid_data = get_grid_1(obstacle_reward=self.hparams['obstacle_reward'],
                                    finish_reward=self.hparams['finish_reward'],
                                    empty_reward=self.hparams['empty_reward'])

        distribution = get_action_distribution(forward_prob=self.hparams['forward_prob'])

        self.mdp = GridMDP(grid=self.grid_data['grid'],
                           action_distribution=distribution,
                           terminals=self.grid_data['terminals'],
                           gamma=self.hparams['gamma'])

        self.utilities, self.policies = self.iteration_algorithm_fn(self.mdp, **self.hparams)

    def get_update_function(self, iteration_slider):
        iteration_slider = iteration_slider

        def update(iteration, **hparams):
            if self.hparams != hparams:
                self.hparams = hparams
                self.get_visualization_data()
                if iteration_slider:
                    iteration_slider.max = len(self.utilities)
                    iteration_slider.value = len(self.utilities)
            else:
                self.visualize(iteration)

        return update

    def visualize(self, iteration):

        fig = plt.figure(figsize=(12, 20), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[self.mdp.cols / self.mdp.rows, 1], figure=fig)
        ax_grid = fig.add_subplot(gs[0, 0])
        ax_distribution = fig.add_subplot(gs[0, 1])

        self.draw_grid_plot(iteration, ax_grid)
        self.draw_action_distribution_plot(ax_distribution)
        plt.show()

    def draw_grid_plot(self, iteration, ax):
        # Define the colors - each entry in the list corresponds to a point in the colormap range
        colors = ["salmon", "white", "lightblue"]

        # Create the colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("utility", colors)
        data = self.utilities[iteration - 1]
        policy = self.policies[iteration - 1]
        grid = []
        grid_P = []
        for row in range(self.mdp.rows):
            current_row_U = []
            current_row_P = []
            for column in range(self.mdp.cols):
                current_row_U.append(data[(column, row)])
                current_row_P.append(policy[column, row])
            grid.append(current_row_U)
            grid_P.append(current_row_P)
        ax.imshow(grid, vmin=self.terminal_reward_min, vmax=self.terminal_reward_max, cmap=cmap,
                  interpolation='nearest')
        ax.axis('off')
        self.draw_surroundings(grid, grid_P, ax)

    def draw_policy(self, iteration, ax):
        data = self.utilities[iteration - 1]
        grid = []
        for row in range(self.mdp.rows):
            current_row = []
            for column in range(self.mdp.cols):
                current_row.append(data[(column, 0, row)])
            grid.append(current_row)

    def draw_surroundings(self, grid, grid_P, ax):
        ax.add_patch(
            patches.Rectangle((self.start_state[0] - 0.5, self.start_state[1] - 0.5), 1, 1,
                              edgecolor='none',
                              facecolor='green', alpha=0.5))
        ax.add_patch(
            patches.Rectangle((self.finish_state[0] - 0.5, self.finish_state[1] - 0.5), 1, 1,
                              edgecolor='none',
                              facecolor='blue', alpha=0.5))
        actions_p = self.actions_path(grid_P)
        for col in range(len(grid)):
            for row in range(len(grid[0])):
                if (col, row) in self.grid_data['obstacles']:
                    rect = patches.Rectangle((row - 0.5, col - 0.5), 1, 1, edgecolor='none',
                                             facecolor='black', alpha=0.5)
                    ax.add_patch(rect)
                else:
                    action = grid_P[col][row]
                    facecolor = 'black'
                    if (row, 0, col) in actions_p:
                        facecolor = 'blue'
                    if action:
                        dx, dy = action[0], action[1]
                        if dx != 0:
                            if dx == -1:
                                arrow = patches.FancyArrow(row - 0.1 * dx, col - 0.25 * dx, 0.1 * action[0],
                                                           0.1 * action[1], width=0.05, edgecolor='none',
                                                           facecolor=facecolor,
                                                           alpha=0.8)
                            else:
                                arrow = patches.FancyArrow(row - 0.1 * dx, col + 0.25 * dx, 0.1 * action[0],
                                                           0.1 * action[1], width=0.05, edgecolor='none',
                                                           facecolor=facecolor,
                                                           alpha=0.8)
                        else:
                            if dy == -1:
                                arrow = patches.FancyArrow(row + 0.1 * dx, col - 0.45 * dy, 0.1 * action[0],
                                                           0.1 * action[1], width=0.05, edgecolor='none',
                                                           facecolor=facecolor,
                                                           alpha=0.8)
                            else:
                                arrow = patches.FancyArrow(row + 0.1 * dx, col + 0.1 * dy, 0.1 * action[0],
                                                           0.1 * action[1], width=0.05, edgecolor='none',
                                                           facecolor=facecolor,
                                                           alpha=0.8)
                        ax.add_patch(arrow)
                value = grid[col][row]
                ax.text(row, col - 0.1, f"{value:.2f}", va='center', ha='center')

    def draw_action_distribution_plot(self, ax):
        ax.imshow([[1]], vmin=0, vmax=1, cmap='gray')

        forwad_dir = UP

        for prob, turn_func in self.mdp.action_distribution:
            facecolor = 'black'
            if turn_func == keep_direction:
                facecolor = 'blue'
            dir = turn_func(forwad_dir)
            dx, dy = dir
            arrow = patches.FancyArrow(0, 0, 0.25 * prob * dx + 0.1 * dx,
                                       0.25 * prob * dy + 0.1 * dy, width=0.025, edgecolor='none',
                                       facecolor=facecolor)
            ax.add_patch(arrow)

        custom_legend_handles = [
            Line2D([0], [1], color='blue', lw=2, marker='>', linestyle='None',
                   label=f"Forward probability: {self.hparams['forward_prob']:.2f}"),
            Line2D([0], [1], color='black', lw=2, marker='>', linestyle='None',
                   label=f"Sideways probability: {(1 - self.hparams['forward_prob']) / 3:.2f}"),
        ]

        # Add the custom legend to the second plot
        ax.legend(handles=custom_legend_handles, loc='lower right')

        ax.axis('off')

    @property
    def start_state(self):
        return self.grid_data['start']

    @property
    def finish_state(self):
        return self.grid_data['finish']

    @property
    def terminals(self):
        return self.grid_data['terminals']


def create_interactive_plot(visualizer, terminal_reward_min, terminal_reward_max):
    visualizer = GridMDPVisualizer(get_grid_fn=get_grid_1,
                                   iteration_algorithm_fn=value_iteration,
                                   terminal_reward_min=terminal_reward_min,
                                   terminal_reward_max=terminal_reward_max,
                                   )

    slider_style = {'description_width': 'initial'}

    iteration_slider = widgets.IntSlider(min=1, max=None, step=1, value=None,
                                         description='Iteration')
    obstacle_reward_slider = widgets.FloatSlider(min=terminal_reward_min, max=0.0, step=0.01, value=-1.0,
                                                 description='Obstacle state reward')
    finish_reward_slider = widgets.FloatSlider(min=0.0, max=terminal_reward_max, step=0.01, value=1.0,
                                               description='Finish state reward')
    empty_reward_slider = widgets.FloatSlider(min=terminal_reward_min, max=terminal_reward_max, step=0.01, value=-0.04,
                                              description='Empty state reward')
    forward_prob_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.8,
                                              description='Forward probability', style=slider_style)
    gamma_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.9, description='Gamma')
    epsilon_slider = widgets.FloatSlider(min=1e-6, max=1e-0, step=1e-6, value=1e-3, description='Epsilon',
                                         readout_format='.2e')
    interactive_plot = widgets.interactive(visualizer.get_update_function(iteration_slider),
                                           iteration=iteration_slider,
                                           obstacle_reward=obstacle_reward_slider,
                                           finish_reward=finish_reward_slider,
                                           empty_reward=empty_reward_slider,
                                           forward_prob=forward_prob_slider,
                                           gamma=gamma_slider,
                                           epsilon=epsilon_slider
                                           )
    display(interactive_plot)


if __name__ == '__main__':
    visualizer = GridMDPVisualizer(get_grid_fn=get_grid_1,
                                   iteration_algorithm_fn=value_iteration,
                                   terminal_reward_min=-1.0,
                                   terminal_reward_max=1.0,
                                   )
    visualizer.get_update_function(None)(iteration=30,
                                         obstacle_reward=-1.0,
                                         finish_reward=1.0,
                                         empty_reward=-0.04,
                                         forward_prob=0.8,
                                         gamma=0.9,
                                         epsilon=1e-3)
