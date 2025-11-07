import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def dist_matrix_calc(mount_df):
    """
    This function calculates the Euclidean distance matrix for all pairs of electrodes
    based on their coordinates in mount_df. The first column is assumed to be the electrode label,
    and the remaining columns are the coordinates (e.g., x_cord, y_cord, z_cord).

    INPUT:
    mount_df = pandas DataFrame with electrode labels and coordinates

    OUTPUT:
    dist_matrix = numpy array of shape (num_channels, num_channels) where each element [r, c]
                  is the Euclidean distance between electrode r and electrode c

    Author: Paulo Roberto Cabral Passos
    ex.: For electrodes at (0,0) and (1,0), the distance is 1.0

    For testing, you can use the following code:
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    try:
      del mount_df
    except:
      print('no variable to be deleted')

    num_channels = 32

    for c in range(num_channels):
      if 'mount_df' not in locals():
        mount_dic = {'channel': [c], 'x_cord': [np.random.uniform(low=-1,high=1)], 'y_cord': [np.random.uniform(low=-1,high=1)] }
        mount_df = pd.DataFrame(mount_dic)
      else:
        mount_dic = {'channel': [c], 'x_cord': [np.random.uniform(low=-1,high=1)], 'y_cord': [np.random.uniform(low=-1,high=1)] }
        new_mount_df = pd.DataFrame(mount_dic)
        mount_df = pd.concat([mount_df, new_mount_df])

    mount_df.reset_index(drop=True)
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    dist_matrix = np.zeros( (mount_df.shape[0], mount_df.shape[0]) )

    for r in range(dist_matrix.shape[0]):
      for c in range(dist_matrix.shape[0]):
        euc_dist = 0
        for p in range(mount_df.shape[1]-1):
          euc_dist = euc_dist + (mount_df.iloc[r, p + 1] - mount_df.iloc[c, p + 1])**2
        euc_dist = np.sqrt(euc_dist)
        dist_matrix[r,c] = euc_dist
    return dist_matrix


def simulating_mount(num_channels):
  """
  """
  try:
    del mount_df
  except:
    print('no variable to be deleted')

  num_channels = 32

  for c in range(num_channels):
    if 'mount_df' not in locals():
      mount_dic = {'channel': [c], 'x_cord': [np.random.uniform(low=-1,high=1)], 'y_cord': [np.random.uniform(low=-1,high=1)] }
      mount_df = pd.DataFrame(mount_dic)
    else:
      mount_dic = {'channel': [c], 'x_cord': [np.random.uniform(low=-1,high=1)], 'y_cord': [np.random.uniform(low=-1,high=1)] }
      new_mount_df = pd.DataFrame(mount_dic)
      mount_df = pd.concat([mount_df, new_mount_df])

  mount_df.reset_index(drop=True)
  return mount_df

def flip_with_probability(binary_vector, flip_probability):
    """
    Flips each entry in a binary vector with a given probability.

    Parameters:
    - binary_vector: List or NumPy array of 0s and 1s.
    - flip_probability: Probability of flipping each entry (float between 0 and 1).

    Returns:
    - A new binary vector with entries flipped according to the given probability.
    """
    # Convert to NumPy array for efficient operations
    vector = np.array(binary_vector)
    # Generate random numbers for each entry
    random_numbers = np.random.random(size=len(vector))
    # Flip entries where random number < flip_probability
    flipped_vector = np.where(random_numbers < flip_probability, 1 - vector, vector)
    return flipped_vector

def calculate_snr(flip_probability):
    """Calculate SNR for binary data given flip probability."""
    return (1 - flip_probability) / flip_probability

def load_and_slice_data(volunteer_data, trial_start, trial_end):
    """
    Loads the data for a volunteer and slices it based on trial_start and trial_end.

    Parameters:
    -----------
    volunteer_data : pandas.DataFrame
        The full data for the volunteer (data_df).
    trial_start : int
        The starting trial index.
    trial_end : int
        The ending trial index.

    Returns:
    --------
    sliced_data : pandas.DataFrame
        The sliced data for the specified trial interval.
    """
    # Slice the data
    sliced_data = volunteer_data.iloc[trial_start:trial_end]

    return sliced_data

# Example usage:
# volunteer_data = pd.read_csv('your_data_file.csv')
# sliced_data = load_and_slice_data(volunteer_data, trial_start=0, trial_end=1000)

def split_data(sliced_data, test_size=0.3, random_state=42):
    """
    Splits the sliced data into training and test sets.

    Parameters:
    -----------
    sliced_data : pandas.DataFrame
        The sliced data for the volunteer.
    test_size : float, optional
        The proportion of the data to include in the test set (default: 0.3).
    random_state : int, optional
        Seed for reproducibility (default: 42).

    Returns:
    --------
    train_data : pandas.DataFrame
        The training set.
    test_data : pandas.DataFrame
        The test set.
    """
    # Split the data
    train_data, test_data = train_test_split(sliced_data, test_size=test_size, random_state=random_state, shuffle=False)

    return train_data, test_data

# Example usage:
# train_data, test_data = split_data(sliced_data)

def estimate_neighborhoods(train_data, c_cte=1.0, max_ne=10):
    """
    Estimates the neighborhood for each choice using the training set.

    Parameters:
    -----------
    train_data : pandas.DataFrame
        The training set.
    c_cte : float, optional
        The penalty constant for the likelihood (default: 1.0).
    max_ne : int, optional
        The maximum number of neighbors (default: 10).

    Returns:
    --------
    neighborhoods : dict
        A dictionary where keys are choice labels and values are the estimated neighborhoods.
    global_summaries : dict
        A dictionary where keys are choice labels and values are the global summaries.
    """
    # Extract EEG features (excluding choice columns)
    eeg_columns = [col for col in train_data.columns if not col.startswith('choice')]
    eeg_train = train_data[eeg_columns].values

    # Initialize dictionaries to store results
    neighborhoods = {}
    global_summaries = {}

    # Loop over each choice
    for choice in ['choice0', 'choice1', 'choice2']:
        # Extract the choice column
        choice_train = train_data[choice].values.reshape(-1, 1)

        # Combine EEG features and the current choice
        input_array = np.hstack([eeg_train, choice_train])

        # Estimate the neighborhood for the current choice (last column)
        ne_matrix, global_summary = m2sdt.get_nematrix(input_array, c_cte, max_ne)

        # Store the results
        neighborhoods[choice] = ne_matrix
        global_summaries[choice] = global_summary

    return neighborhoods, global_summaries

# Example usage:
# neighborhoods, global_summaries = estimate_neighborhoods(train_data)
