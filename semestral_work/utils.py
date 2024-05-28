import numpy as np

orientations = RIGHT, UP, LEFT, DOWN = [(1, 0), (0, 1), (-1, 0), (0, -1)]


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


def get_grid_1(obstacle_reward, finish_reward, empty_reward):
    O, F, S, _ = obstacle_reward, finish_reward, empty_reward, empty_reward
    grid = [[_, _, _, O, O, _, _, _, _, _, _, O, _],
            [S, _, _, O, O, _, _, _, _, _, _, O, _],
            [_, _, _, _, O, _, _, _, O, _, _, _, _],
            [_, _, _, _, O, _, _, _, O, _, _, _, _],
            [_, O, _, _, _, _, _, O, O, O, _, _, F],
            [_, O, _, _, _, _, _, O, O, O, _, _, _]]

    # positions are given as (x, y) pairs, where origin (0, 0) is in the lower left corner of grid.
    obstacles = [(1, 0), (1, 1),
                 (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5),
                 (7, 0), (7, 1), (8, 0), (8, 1), (8, 2), (8, 3), (9, 0), (9, 1),
                 (11, 4), (11, 5)]
    finish = (12, 1)
    terminals = obstacles.append(finish)
    start = (0, 4)
    return grid, start, obstacles, finish, terminals


if __name__ == '__main__':
    # Example usage:
    vector = (1, 0)
    print(f"Original vector: {vector}")
    angle_degree = 90
    rotated_vector = rotate_vector_2d(vector, angle_degree)
    print("Rotated vector:", rotated_vector, "by", angle_degree)
    direction = vector_to_direction(rotated_vector)
    print("Direction:", direction)