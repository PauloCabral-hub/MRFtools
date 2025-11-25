import itertools
import numpy as np
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc
import random
from scipy import sparse
from self_dev_tools import module1sdt as m1sdt
from self_dev_tools import module2sdt as m2sdt
from self_dev_tools import module3sdt as m3sdt


def get_cand_tillsize(node, V_set, max_neighborhood=10, max_candidates=1000):
    """
    Generates a limited number of candidate neighborhoods for a given node.

    Parameters:
    -----------
    node : int
        The node for which to generate candidate neighborhoods.
    V_set : set
        The set of all nodes in the system.
     max_candidates : int, optional
        Maximum number of candidate neighborhoods to generate (default: 1000).

    Returns:
    --------
    candidates : list of lists
        A limited list of candidate neighborhoods.
    """
    candidates = []
    n_nodes = len(V_set) - 1  # Exclude the node itself

    # Limit the maximum neighborhood size to reduce combinatorial explosion
    max_size = min(max_neighborhood, n_nodes)  

    for csize in range(1, max_size + 1):
        # Pass node as a set to get_candidates
        new_candidates = get_candidates({node}, V_set, csize)
        new_candidates = list(new_candidates)

        # Randomly sample candidates if there are too many
        if len(new_candidates) > max_candidates:
            new_candidates = random.sample(new_candidates, max_candidates)

        candidates.extend(new_candidates)

    return candidates


def get_candidates(node, V_set, cdate_size):
    """
    Generates all candidate neighborhoods of a specific size for a given node.

    Parameters:
    -----------
    node : set
        A set containing the node for which to generate candidate neighborhoods.
    V_set : set
        The set of all nodes in the system.
    cdate_size : int
        The size of the candidate neighborhoods to generate.

    Returns:
    --------
    cdate_list : numpy.ndarray
        An array of candidate neighborhoods.
    """
    nhood = V_set.difference(node)
    if cdate_size > len(nhood):
        print('<cdate_size> cannot be greater than the cardinal of V - \{ node \}')
        cdate_list = list()
        return cdate_list
    cdate_list = np.array(list(itertools.combinations(list(nhood), cdate_size)))
    return cdate_list

def node_cdnhood(cdnhood_size):
    """
    Generates all possible configurations of a node and its candidate neighborhood.

    This function creates a matrix where each row represents a unique configuration of a node and its
    neighborhood of size `cdnhood_size`. The first column corresponds to the state of the node (0 or 1),
    and the remaining columns correspond to the states of the neighborhood nodes. This is used to enumerate
    all possible joint states of a node and its neighborhood.

    Parameters:
    -----------
    cdnhood_size : int
        The size of the candidate neighborhood.

    Returns:
    --------
    ncdnhood_config : numpy.ndarray
        A 2D array where each row is a unique configuration of the node and its neighborhood.
        For example, if cdnhood_size = 1, the output is:
        [[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]]
        where the first column is the node's state, and the second column is the state of its single neighbor.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    ncdnhood_config = np.array(list(itertools.product([0, 1],repeat=cdnhood_size + 1)))

    return ncdnhood_config

def count_ncdnhood(joint_configs, realizations):
    """
    Counts the occurrences of each node-neighborhood configuration in the data.

    This function takes a matrix of all possible joint configurations (node + neighborhood) and a matrix
    of observed realizations (data), then counts how many times each configuration appears in the data.
    It also counts how many times each neighborhood configuration appears, regardless of the node's state.

    Parameters:
    -----------
    joint_configs : numpy.ndarray
        A matrix where each row is a unique joint configuration of a node and its neighborhood.
    realizations : numpy.ndarray
        A matrix of observed data, where each row is a realization (sample) of the node and its neighborhood.

    Returns:
    --------
    cdnhood_configs : numpy.ndarray
        Unique neighborhood configurations (excluding the node's state).
    cdnhood_count : numpy.ndarray
        The count of each neighborhood configuration in the data.
    joint_configs : numpy.ndarray
        Unique joint configurations of the node and its neighborhood.
    joint_count : numpy.ndarray
        The count of each joint configuration in the data.
    cdconfig_map : numpy.ndarray
        A mapping from each joint configuration to its corresponding neighborhood configuration.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    # Counting a_w's
    if joint_configs.shape[1] == 1:
      cdnhood_configs = []
      cdnhood_count = [ realizations.shape[0] ];
    else:
      cdnhood_configs = np.unique(joint_configs[:,1:],axis=0)
      cdnhood_count = np.zeros(cdnhood_configs.shape[0])
      for c in range(cdnhood_configs.shape[0]):
        cdnhood_count[c] = (realizations[:,1:] == cdnhood_configs[c,:]).all(axis=1).sum()

    # Counting a_v's AND a_w's

    joint_count = np.zeros(joint_configs.shape[0])
    cdconfig_map = np.zeros(joint_configs.shape[0])
    for c in range( joint_configs.shape[0] ):
        joint_count[c] = (realizations == joint_configs[c,:]).all(axis=1).sum()

        if joint_configs.shape[1] > 1:
          check_board = (cdnhood_configs == joint_configs[c,1:]).all(axis=1)
          cdconfig_map[c] = np.where(check_board)[0][0]
        else:
          cdconfig_map[c] = 0

    return cdnhood_configs, cdnhood_count, joint_configs, joint_count, cdconfig_map


def associate_cdate(input_array, node, cdate):
    """
    Associates a candidate neighborhood with a node and computes statistics for their joint configurations.

    Given an input matrix of realizations (samples), a node, and a candidate neighborhood, this function
    computes the joint configurations of the node and its neighborhood, and counts how often each configuration
    appears in the data. This is a key step in estimating the likelihood of a candidate neighborhood.

    Parameters:
    -----------
    input_array : numpy.ndarray
        A matrix of realizations, where each row is a sample and each column is a node.
    node : int
        The node of interest.
    cdate : list
        The candidate neighborhood (list of node indices).

    Returns:
    --------
    associate_summary : dict
        A dictionary containing:
        - 'node': The node of interest.
        - 'cdate': The candidate neighborhood.
        - 'cdnhood_configs': Unique neighborhood configurations.
        - 'cdnhood_count': Counts of each neighborhood configuration.
        - 'joint_configs': Unique joint configurations of the node and its neighborhood.
        - 'joint_count': Counts of each joint configuration.
        - 'cdconfig_map': Mapping from joint configurations to neighborhood configurations.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    #in:
    if len(cdate) == 0:
      scope_array = input_array[:,[node]]
    else:
    #in:
      scope_array = input_array[:,[node] + cdate]
    ncdnhood_config = node_cdnhood(len(cdate))
    cdnhood_configs, cdnhood_count, joint_configs, joint_count, cdconfig_map = count_ncdnhood(ncdnhood_config, scope_array)
    associate_summary = {'node': node, 'cdate': cdate, 'cdnhood_configs': cdnhood_configs, 'cdnhood_count': cdnhood_count,
                         'joint_configs': joint_configs, 'joint_count': joint_count, 'cdconfig_map': cdconfig_map}
    return associate_summary

def calculate_likelihood(associate_summary):
  """
  Calculates the likelihood of a node's state given its candidate neighborhood.

  This function computes the likelihood of observing the data under the assumption that the candidate
  neighborhood is the true neighborhood. It calculates the conditional probability of the node's state
  given each neighborhood configuration and computes the overall likelihood as the product of these probabilities.

  Parameters:
  -----------
  associate_summary : dict
      The output of `associate_cdate`, containing joint configurations and their counts.

  Returns:
  --------
  associate_summary : dict
      The input dictionary, updated with:
      - 'probabilities': Conditional probabilities of the node's state given each neighborhood configuration.
      - 'll_hood': The overall likelihood of the data given the candidate neighborhood.

  Author: Paulo Roberto Cabral Passos
  Date: October 20, 2025
  """
  ll_hood = 1
  associate_summary['probabilities'] = np.zeros(associate_summary['joint_configs'].shape[0])
  for c in range(associate_summary['joint_configs'].shape[0]):
    if associate_summary['joint_count'][c] != 0:
      p_num = associate_summary['joint_count'][c]
      p_den = associate_summary['cdnhood_count'][  int(associate_summary['cdconfig_map'][c])    ]
      associate_summary['probabilities'][c] = p_num/p_den
      p = (p_num/p_den)**p_num
      ll_hood = ll_hood*p
  associate_summary['ll_hood'] = ll_hood
  return associate_summary

def penalize_llhood(associate_summary, c, N):
    """
    Applies a penalty to the likelihood to avoid overfitting.

    This function adjusts the log-likelihood by penalizing larger neighborhoods. The penalty is proportional
    to the size of the neighborhood and the number of realizations, which helps prevent the selection of
    overly complex neighborhoods that might fit the data by chance.

    Parameters:
    -----------
    associate_summary : dict
        The output of 'calculate_likelihood', containing the likelihood of the candidate neighborhood.
    c : float
        The penalty constant, controlling the strength of the penalty.
    N : int
        The number of realizations (samples) in the data.

    Returns:
    --------
    associate_summary : dict
        The input dictionary, updated with:
        - 'pen_llhood': The penalized log-likelihood of the candidate neighborhood.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    associate_summary['pen_llhood'] = np.log(associate_summary['ll_hood']) - c*np.log(N)*2**len(associate_summary['cdate']) # Assuming |A| = 2
    
    return associate_summary


def filt_candidates(all_candidates, max_ne):
    """
    Filters candidate neighborhoods to enforce a maximum size constraint.

    This function removes any candidate neighborhoods that exceed the maximum allowed size (`max_ne`).
    This ensures that only neighborhoods of feasible size are considered in the analysis.

    Parameters:
    -----------
    all_candidates : list
        A list of candidate neighborhoods.
    max_ne : int
        The maximum allowed neighborhood size.

    Returns:
    --------
    filt_candidates : list
        A filtered list of candidate neighborhoods, where each neighborhood has size <= max_ne.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    filt_candidates = []
    for i in range( len(all_candidates) ):
        if type(all_candidates[i]) ==  list or all_candidates[i].shape[0] <= max_ne:
            filt_candidates.append(all_candidates[i])
    return filt_candidates

def get_nematrix(input_array, c_cte, max_ne):
    """
    Estimates the neighborhood structure for all nodes in the system.

    Given a matrix of realizations, a penalty constant, and a maximum neighborhood size, this function
    estimates the neighborhood for each node by selecting the candidate neighborhood that maximizes the
    penalized likelihood. It returns an adjacency matrix representing the neighborhood structure and a
    dictionary of summaries for each node's neighborhood.

    Parameters:
    -----------
    input_array : numpy.ndarray
        A matrix of realizations, where each row is a sample and each column is a node.
    c_cte : float
        The penalty constant for the likelihood.
    max_ne : int
        The maximum allowed neighborhood size.

    Returns:
    --------
    ne_matrix : numpy.ndarray
        A directed adjacency matrix where ne_matrix[i, j] = 1 if node j is in the neighborhood of node i.
    global_summary : dict
        A dictionary where each key is a node, and the value is the summary of its estimated neighborhood.

    Author: Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    # node loop:
    ne_matrix = np.zeros( (input_array.shape[1],input_array.shape[1]) )
    #insert>
    global_summary = dict()
    #insert<
    for node_i in range(input_array.shape[1]):
        winner_ne = []
        winner_score = -10**100
        #insert>
        winner_summary = {}
        #insert<
        all_candidates = m2sdt.get_cand_tillsize({node_i}, set(range(input_array.shape[1])))
        all_candidates.insert(0, [])
        #insert>
        all_candidates = filt_candidates(all_candidates, max_ne)        
        #insert<
        # candidate loop
        for cdate in all_candidates:
            associate_summary = m2sdt.associate_cdate(input_array, node_i, cdate.tolist() if len(cdate) > 0 else cdate)
            associate_summary = m2sdt.calculate_likelihood(associate_summary)
            associate_summary = m2sdt.penalize_llhood(associate_summary, c_cte, input_array.shape[0])
            if associate_summary['pen_llhood'] > winner_score:
                winner_score = associate_summary['pen_llhood']
                winner_ne = associate_summary['cdate']
                #insert>
                winner_summary = associate_summary
                #insert<
        #insert>
        global_summary[winner_summary['node']] = winner_summary
        #insert<
        for node_j in range(input_array.shape[1]):
            for item in winner_ne:
                if node_j == item:
                    ne_matrix[node_i][node_j] = 1
    return ne_matrix, global_summary


def estimate_neighborhoods(train_data, c_cte=0.1, max_ne=10, chunk_size=200, max_candidates=1000):
    """
    Estimates the neighborhood for each choice using the training set in chunks,
    with explicit garbage collection and limited candidate space to manage memory usage.
    !! ATTENTION: All choice nodes are labeled 32 !!

    Parameters:
    -----------
    train_data : pandas.DataFrame
        The training set.
    c_cte : float, optional
        The penalty constant for the likelihood (default: 1.0).
    max_ne : int, optional
        The maximum number of neighbors (default: 10).
    chunk_size : int, optional
        Number of trials to process at a time (default: 200).
    max_candidates : int, optional
        Maximum number of candidate neighborhoods to generate (default: 1000).

    Returns:
    --------
    neighborhoods : dict
        A dictionary where keys are choice labels and values are the estimated neighborhoods.
    global_summaries : dict
        A dictionary where keys are choice labels and values are the global summaries.
    """

    # Extract EEG features
    eeg_columns = [col for col in train_data.columns if not col.startswith('choice')]

    # Initialize dictionaries to store results
    neighborhoods = {}
    global_summaries = {}

    # Loop over each choice with a progress bar

    for choice in tqdm(['choice0', 'choice1', 'choice2'], desc="Estimating neighborhoods"):
        chunked_input_arrays = []
        # Process data in chunks
        for start in tqdm(range(0, len(train_data), chunk_size), desc=f"Processing {choice} in chunks", leave=False):
            end = start + chunk_size
            chunk = train_data.iloc[start:end]

            # Convert to float32 to save memory
            eeg_chunk = chunk[eeg_columns].values.astype(np.float32)
            choice_chunk = chunk[choice].values.reshape(-1, 1).astype(np.float32)

            input_array_chunk = np.hstack([eeg_chunk, choice_chunk])
            chunked_input_arrays.append(input_array_chunk)

            # Explicitly delete intermediate variables and collect garbage
            del eeg_chunk, choice_chunk
            gc.collect()

        # Concatenate chunks into a single input array
        input_array = np.vstack(chunked_input_arrays).astype(np.float32)

        # Explicitly delete chunked_input_arrays and collect garbage
        del chunked_input_arrays
        gc.collect()

        # Estimate the neighborhood for the current choice
        # Get the index of the choice node (last column)
        node_i = input_array.shape[1] - 1

        # Get candidate neighborhoods with limited candidates
        all_candidates = get_cand_tillsize(node_i, set(range(input_array.shape[1])), max_neighborhood=max_ne, max_candidates=max_candidates)
        all_candidates.insert(0, [])  # Include the empty neighborhood

        # Filter candidates based on max_ne
        all_candidates = filt_candidates(all_candidates, max_ne)

        # Initialize variables for the best neighborhood
        winner_ne = []
        winner_score = -10**100
        winner_summary = {}

        # Loop over candidate neighborhoods
        for cdate in all_candidates:
            associate_summary = associate_cdate(input_array, node_i, list(cdate) if len(cdate) > 0 else cdate)
            associate_summary = calculate_likelihood(associate_summary)
            associate_summary = penalize_llhood(associate_summary, c_cte, input_array.shape[0])

            if associate_summary['pen_llhood'] > winner_score:
                winner_score = associate_summary['pen_llhood']
                winner_ne = associate_summary['cdate']
                winner_summary = associate_summary

        # Store the results
        # Create adjacency matrix
        ne_matrix = np.zeros((input_array.shape[1], input_array.shape[1]))
        for node_j in range(input_array.shape[1]):
            for item in winner_ne:
                if node_j == item:
                    ne_matrix[node_i][node_j] = 1

        neighborhoods[choice] = ne_matrix
        global_summaries[choice] = winner_summary

        # Explicitly delete intermediate variables and collect garbage
        del input_array, all_candidates, winner_ne, winner_summary
        gc.collect()

    return neighborhoods, global_summaries