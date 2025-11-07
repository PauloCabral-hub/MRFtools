import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_splash(chan_info):
    """
    Given a DataFrame of channel information, returns a matrix with the 2D projection
    of the 3D location of the channels, adjusted to prevent overlap.

    Parameters:
    -----------
    chan_info : pandas.DataFrame
        DataFrame with columns 'chan_name', 'x_coor', 'y_coor', and 'z_coor'.

    Returns:
    --------
    channels : list of str
        List of channel labels.
    splash : numpy.ndarray
        Matrix with the 2D projection of the 3D channel locations.
    """
    channels = []
    V = np.zeros((len(chan_info), 3))

    # Find Cz coordinates (assuming Cz is the 32nd channel)
    Cz_coord = np.array([chan_info['x_coor'].iloc[31], chan_info['y_coor'].iloc[31], 0])

    # Centralize coordinates in Cz
    for a in range(len(chan_info)):
        channels.append(chan_info['chan_name'].iloc[a])
        V[a, 0] = chan_info['x_coor'].iloc[a] - Cz_coord[0]
        V[a, 1] = chan_info['y_coor'].iloc[a] - Cz_coord[1]
        V[a, 2] = chan_info['z_coor'].iloc[a]

    # Make the minimum Z equal to 0
    V[:, 2] = V[:, 2] + abs(np.min(V[:, 2]))

    # Get the maximum height
    h_max = np.max(V[:, 2])

    splash = np.zeros((V.shape[0], 2))

    for a in range(len(chan_info)):
        v_p = V[a, 0:2]
        if np.linalg.norm(v_p) != 0:
            u_p = v_p / np.linalg.norm(v_p)
        else:
            u_p = np.zeros(2)

        splash[a, :] = v_p + u_p * abs(h_max - V[a, 2])

        # Rotate the vector by 90 degrees counterclockwise
        splash[a, :] = np.rot90(v_p.reshape(1, -1), k=1).flatten()

    return channels, splash


def add_choice_nodes(channels, splash, choice_labels):
    """
    Adds choice nodes to the splash matrix, placing them horizontally at the top center of the plot.

    Parameters:
    -----------
    channels : list of str
        List of EEG channel labels.
    splash : numpy.ndarray
        Matrix with the 2D projection of the 3D channel locations.
    choice_labels : list of str
        List of choice labels (e.g., ['choice0', 'choice1', 'choice2']).

    Returns:
    --------
    all_channels : list of str
        List of all channel labels, including choices.
    all_splash : numpy.ndarray
        Matrix with the 2D projection of all channel locations, including choices.
    """
    # Calculate the diameter of the circle encompassing the EEG electrodes
    x_coords = splash[:, 0]
    y_coords = splash[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    diameter = np.max([x_max - x_min, y_max - y_min])
    spacing = diameter / 4

    # Center x-coordinate for choice nodes
    center_x = np.mean([x_min, x_max])

    # Top y-coordinate for choice nodes (above the highest EEG electrode)
    top_y = y_max + spacing

    # Calculate x-coordinates for choice nodes (horizontally aligned)
    num_choices = len(choice_labels)
    choice_x_coords = np.linspace(center_x - (num_choices - 1) * spacing / 2,
                                   center_x + (num_choices - 1) * spacing / 2,
                                   num_choices)

    # Add choice nodes
    all_channels = channels + choice_labels
    all_splash = np.vstack([splash, np.zeros((len(choice_labels), 2))])

    for i, label in enumerate(choice_labels):
        all_splash[len(channels) + i, 0] = choice_x_coords[i]
        all_splash[len(channels) + i, 1] = top_y

    return all_channels, all_splash