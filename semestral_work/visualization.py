from matplotlib.lines import Line2D

from mdp import GridMDP, value_iteration
from utils import get_action_distribution, get_grid, grid_1, keep_direction, actions_path
from utils import UP
import ipywidgets as widgets
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


class GridMDPVisualizer:
    def __init__(self, grid_structure_fn=grid_1,
                 iteration_algorithm_fn=value_iteration,
                 *,
                 terminal_reward_min=-10.0,
                 terminal_reward_max=10.0):

        self.grid_structure_fn = grid_structure_fn
        self.iteration_algorithm_fn = iteration_algorithm_fn

        self.terminal_reward_min = terminal_reward_min
        self.terminal_reward_max = terminal_reward_max

        self.grid_data = None
        self.hparams = None
        self.mdp = None
        self.utilities = None
        self.policies = None
        self.current_iteration = None

    def get_visualization_data(self):
        self.grid_data = get_grid(obstacle_reward=self.hparams['obstacle_reward'],
                                  finish_reward=self.hparams['finish_reward'],
                                  empty_reward=self.hparams['empty_reward'],
                                  grid_structure_fn=self.grid_structure_fn)

        distribution = get_action_distribution(forward_prob=self.hparams['forward_prob'])

        self.mdp = GridMDP(grid=self.grid_data['grid'],
                           action_distribution=distribution,
                           terminals=self.grid_data['terminals'],
                           gamma=self.hparams['gamma'])

        self.utilities, self.policies = self.iteration_algorithm_fn(self.mdp, **self.hparams)

    def iteration_plot(self, iteration, **hparams):
        self.hparams = hparams
        self.get_visualization_data()
        self.current_iteration = iteration
        self.visualize()

    def get_update_function(self, iteration_slider):
        iteration_slider = iteration_slider

        def update(iteration, **hparams):
            if self.hparams != hparams:
                self.hparams = hparams
                self.get_visualization_data()
                self.current_iteration = len(self.utilities)
                iteration_slider.max = len(self.utilities)
                iteration_slider.value = len(self.utilities)
                self.visualize()
            elif self.current_iteration != iteration:
                self.current_iteration = iteration
                self.visualize()

        return update

    def visualize(self):

        fig = plt.figure(figsize=(12, 20), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[self.mdp.cols / self.mdp.rows, 1], figure=fig)
        ax_grid = fig.add_subplot(gs[0, 0])
        ax_distribution = fig.add_subplot(gs[0, 1])

        self.draw_grid_plot(ax_grid)
        self.draw_action_distribution_plot(ax_distribution)
        plt.show()

    def draw_grid_plot(self, ax):

        utility = self.utilities[self.current_iteration - 1]
        policy = self.policies[self.current_iteration - 1]
        self.draw_utility(utility, ax)
        self.draw_surroundings(ax)
        self.draw_utility_text(utility, ax)
        self.draw_policy(policy, ax)
        ax.axis('off')
        # self.draw_surroundings(grid, grid_P, ax)

    def draw_utility(self, utility, ax):
        colors = ["salmon", "white", "lightblue"]
        cmap = mcolors.LinearSegmentedColormap.from_list("utility", colors)
        utility_grid = []
        for row in range(self.mdp.rows):
            current_row = []
            for column in range(self.mdp.cols):
                current_row.append(utility[(column, row)])
            utility_grid.append(current_row)
        ax.imshow(utility_grid, vmin=self.terminal_reward_min, vmax=self.terminal_reward_max, cmap=cmap,
                  interpolation='nearest', origin='lower')

    def draw_surroundings(self, ax):
        # draw start state
        self.draw_rect(self.grid_data['start'], "green", ax)
        # draw finish state
        self.draw_rect(self.grid_data['finish'], "blue", ax)
        # draw obstacles
        for obstacle_pos in self.grid_data['obstacles']:
            self.draw_rect(obstacle_pos, "black", ax)

    def draw_utility_text(self, utility, ax):
        for state, value in utility.items():
            x, y = state
            ax.text(x, y + (0.1 if state not in self.grid_data['terminals'] else 0.0),
                    f"{value:.2f}", va='center', ha='center')

    @staticmethod
    def draw_rect(position, color, ax, alpha=0.5):
        rectangle = patches.Rectangle((position[0] - 0.5, position[1] - 0.5), 1, 1, facecolor=color, alpha=alpha)
        ax.add_patch(rectangle)

    def draw_policy(self, policy, ax):
        path = actions_path(self.grid_data['start'], policy)
        for policy_pos, direction in policy.items():
            if direction is None:
                continue

            if policy_pos in path:
                facecolor = 'blue'
            else:
                facecolor = 'black'

            x, y = policy_pos
            dx, dy = direction

            if dx != 0:
                self.draw_arrow(x - 0.1 * dx, y - 0.25 * abs(dx), 0.1 * dx, 0, facecolor, ax)
            else:
                if dy == -1:
                    self.draw_arrow(x, y - 0.1 * abs(dy), 0,
                                    0.1 * dy, facecolor, ax)
                else:
                    self.draw_arrow(x, y - 0.4 * dy, 0,
                                    0.1 * dy, facecolor, ax)

    def draw_arrow(self, x, y, dx, dy, facecolor, ax):
        arrow = patches.FancyArrow(x, y, dx, dy, width=0.05, edgecolor='none', facecolor=facecolor, alpha=0.8)
        ax.add_patch(arrow)

    def draw_action_distribution_plot(self, ax):
        # draw empty square
        ax.imshow([[1]], vmin=0, vmax=1, cmap='gray', origin='lower')

        forwad_dir = UP

        for prob, turn_func in self.mdp.action_distribution:
            if turn_func == keep_direction:
                facecolor = 'blue'
            else:
                facecolor = 'black'

            dir = turn_func(forwad_dir)
            dx, dy = dir
            hard_offset_x, hard_offset_y = 0.1 * dx, 0.1 * dy
            arrow = patches.FancyArrow(0, 0, 0.25 * prob * dx + hard_offset_x,
                                       0.25 * prob * dy + hard_offset_y, width=0.025, edgecolor='none',
                                       facecolor=facecolor)
            ax.add_patch(arrow)

        custom_legend_handles = [
            Line2D([0], [1], color='blue', lw=2, marker='>', linestyle='None',
                   label=f"Forward probability: {self.hparams['forward_prob']:.2f}"),
            Line2D([0], [1], color='black', lw=2, marker='>', linestyle='None',
                   label=f"Sideways probability: {(1 - self.hparams['forward_prob']) / 3:.2f}"),
        ]

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


def create_interactive_plot(grid_fn, terminal_reward_min=-1.0, terminal_reward_max=1.0):
    visualizer = GridMDPVisualizer(grid_structure_fn=grid_fn,
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
    visualizer = GridMDPVisualizer(grid_structure_fn=grid_1,
                                   iteration_algorithm_fn=value_iteration,
                                   terminal_reward_min=-1.0,
                                   terminal_reward_max=1.0,
                                   )
    visualizer.iteration_plot(iteration=31,
                              obstacle_reward=-1.0,
                              finish_reward=1.0,
                              empty_reward=-0.04,
                              forward_prob=0.8,
                              gamma=1.0,
                              epsilon=1e-3)
