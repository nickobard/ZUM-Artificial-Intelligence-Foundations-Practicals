from collections import deque

import numpy as np
import math

orientations = RIGHT, UP, LEFT, DOWN = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def add_tuple_vectors(v1, v2):
    """Return the state that results from going in this direction."""
    if isinstance(v1, tuple):
        v1 = np.array(v1)
    if isinstance(v2, tuple):
        v2 = np.array(v2)
    result = tuple(v1 + v2)
    return result


def rotate_vector_2d(vector, angle_degree):
    if isinstance(vector, tuple):
        vector = np.array(vector)
    angle_rad = np.radians(angle_degree)
    # Rotation matrix for 2D vectors
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    # Apply the rotation
    result = np.dot(rot_matrix, vector)
    return result


def normalize_vector(vector):
    # Compute the Euclidean norm (magnitude) of the vector
    magnitude = np.linalg.norm(vector)

    # Avoid division by zero
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")

    # Divide each component by the magnitude to get the normalized vector
    normalized_vector = vector / magnitude
    return normalized_vector


def vector_to_direction(vector):
    return np.round(normalize_vector(vector)).astype(int)


def keep_direction(vector):
    return np.array(vector)


def turn_up(vector):
    # Rotate -90 degrees around the y-axis
    rotated_vector = rotate_vector_2d(vector, 90)
    direction = vector_to_direction(rotated_vector)
    return direction


def turn_down(vector):
    # Rotate 90 degrees around the y-axis
    rotated_vector = rotate_vector_2d(vector, -90)
    direction = vector_to_direction(rotated_vector)
    return direction


def turn_backward(vector):
    # Rotate 180 degrees around the y-axis
    return -np.array(vector)


def get_action_distribution(forward_prob):
    distributions = [(forward_prob, keep_direction)]
    turn_actions = [turn_up, turn_down, turn_backward]

    complement_prob = 1 - forward_prob
    distributions.extend(((complement_prob / len(turn_actions), action) for action in turn_actions))
    return distributions


def actions_path(start, policy):
    visited = {start}
    opened = deque([start])
    states = policy.keys()
    while opened:
        state = opened.pop()
        action = policy[state]
        if action is None:
            continue
        next_state = add_tuple_vectors(state, action)
        if next_state in states and next_state not in visited:
            visited.add(next_state)
            opened.append(next_state)
    return visited


def grid_1(O, F, S, _):
    return [[_, _, _, O, O, _, _, _, _, _, _, O, _],
            [S, _, _, O, O, _, _, _, _, _, _, O, _],
            [_, _, _, _, O, _, _, _, O, _, _, _, _],
            [_, _, _, _, O, _, _, _, O, _, _, _, _],
            [_, O, _, _, _, _, _, O, O, O, _, _, F],
            [_, O, _, _, _, _, _, O, O, O, _, _, _]]


def get_grid(obstacle_reward, finish_reward, empty_reward, grid_structure_fn=grid_1):
    O, F, S, _ = (("obstacle", obstacle_reward),
                  ("finish", finish_reward),
                  ("start", empty_reward),
                  ("empty", empty_reward))
    variables_grid = grid_structure_fn(O, F, S, _)
    return parse_grid(variables_grid)


def parse_grid(grid_variables):
    grid = []
    obstacles = []
    finish = None
    start = None
    grid_variables_reversed_rows = grid_variables[::-1]
    # positions are given as (x, y) pairs, where origin (0, 0) is in the lower left corner of grid.
    for y in range(len(grid_variables_reversed_rows)):
        current_row = []
        for x in range(len(grid_variables_reversed_rows[y])):
            state_type, state_reward = grid_variables_reversed_rows[y][x]
            if state_type == "obstacle":
                obstacles.append((x, y))
            elif state_type == "finish":
                finish = (x, y)
            elif state_type == "start":
                start = (x, y)
            current_row.append(state_reward)
        grid.append(current_row)

    terminals = obstacles.copy()
    terminals.append(finish)

    return {'grid': grid,
            'start': start,
            'obstacles': obstacles,
            'finish': finish,
            'terminals': terminals}


if __name__ == '__main__':
    vector = (1, 0)
    print(f"Original vector: {vector}")
    angle_degree = 90
    rotated_vector = rotate_vector_2d(vector, angle_degree)
    print("Rotated vector:", rotated_vector, "by", angle_degree)
    direction = vector_to_direction(rotated_vector)
    print("Direction:", direction)

    print(50 * '-')

    print(get_action_distribution(0.8))

    assert math.isclose(sum((p for (p, _) in get_action_distribution(0.8))),
                        1.0, rel_tol=1e-9)
    assert math.isclose(sum((p for (p, _) in get_action_distribution(0.1))),
                        1.0, rel_tol=1e-9)

    print(50 * '-')
