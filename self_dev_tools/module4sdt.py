import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import itertools
from collections import defaultdict
#
from self_dev_tools import module1sdt as m1sdt
from self_dev_tools import module2sdt as m2sdt
from self_dev_tools  import module3sdt as m3sdt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def simulate_and_test(flip_probabilities, penalty_constants, n_repeats=50, n_samples=500):
    """
    Simulates neighborhood detection for a single noisy neighbor and evaluates performance.

    This function tests whether the penalized likelihood method can correctly identify a single noisy neighbor
    of a node. For each combination of flip probability (noise level) and penalty constant, it generates
    synthetic data (a node and its noisy copy), then compares the likelihood of the empty neighborhood
    versus the true neighborhood (the noisy copy). The proportion of correct identifications is recorded
    for each combination of parameters.

    Parameters:
    -----------
    flip_probabilities : list of float
        List of probabilities for flipping bits (noise levels).
    penalty_constants : list of float
        List of penalty constants for the likelihood.
    n_repeats : int, optional
        Number of repetitions for each parameter combination (default: 50).
    n_samples : int, optional
        Number of samples (realizations) to generate (default: 500).

    Returns:
    --------
    results : defaultdict of list of tuples
        A dictionary where keys are penalty constants, and values are lists of tuples (SNR, proportion_correct).
        Each tuple contains the signal-to-noise ratio (SNR) and the proportion of correct identifications.

    Author: AI generated with orientation of Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    results = defaultdict(list)

    for c in penalty_constants:
        for p in flip_probabilities:
            correct_count = 0

            for _ in range(n_repeats):
                # 1. Generate v and u (noisy version of v)
                v = np.random.randint(0, 2, n_samples)
                u = m3sdt.flip_with_probability(v, p)

                # 2. Calculate likelihoods
                # Empty neighborhood
                empty_summary = m2sdt.associate_cdate(np.column_stack([v, u]), 0, [])
                empty_summary = m2sdt.calculate_likelihood(empty_summary)
                empty_summary = m2sdt.penalize_llhood(empty_summary, c, n_samples)

                # Neighborhood [u]
                neighbor_summary = m2sdt.associate_cdate(np.column_stack([v, u]), 0, [1])
                neighbor_summary = m2sdt.calculate_likelihood(neighbor_summary)
                neighbor_summary = m2sdt.penalize_llhood(neighbor_summary, c, n_samples)

                # 3. Determine if u is a neighbor
                if neighbor_summary['pen_llhood'] > empty_summary['pen_llhood']:
                    correct_count += 1

            # Store proportion of correct identifications
            snr = m3sdt.calculate_snr(p)
            results[c].append((snr, correct_count / n_repeats))

    return results

def simulate_and_test2(flip_probabilities, penalty_constants, max_neighbors=8, n_repeats=50, n_samples=500):
    """
    Simulates neighborhood detection for varying neighborhood sizes and evaluates performance.

    This function tests the penalized likelihood method's ability to correctly identify neighborhoods
    of different sizes. For each combination of flip probability, penalty constant, and neighborhood size,
    it generates synthetic data (a node and multiple noisy copies), then compares the likelihood of the
    empty neighborhood versus the true neighborhood (a subset of the noisy copies). The proportion of
    correct identifications is recorded for each combination of parameters.

    Parameters:
    -----------
    flip_probabilities : list of float
        List of probabilities for flipping bits (noise levels).
    penalty_constants : list of float
        List of penalty constants for the likelihood.
    max_neighbors : int, optional
        Maximum number of neighbors to test (default: 8).
    n_repeats : int, optional
        Number of repetitions for each parameter combination (default: 50).
    n_samples : int, optional
        Number of samples (realizations) to generate (default: 500).

    Returns:
    --------
    results : dict of defaultdict of list of tuples
        A nested dictionary where the outer key is the neighborhood size, the inner key is the penalty constant,
        and the value is a list of tuples (SNR, proportion_correct). Each tuple contains the signal-to-noise
        ratio (SNR) and the proportion of correct identifications.

    Author: AI generated with orientation of Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    results = {size: defaultdict(list) for size in range(1, max_neighbors + 1)}

    for neighbor_size in range(1, max_neighbors + 1):
        for c in penalty_constants:
            for p in flip_probabilities:
                correct_count = 0

                for _ in range(n_repeats):
                    # 1. Generate v and and noisy copies
                    v = np.random.randint(0, 2, n_samples)
                    u_copies = [m3sdt.flip_with_probability(v, p) for _ in range(max_neighbors)]
                    data = np.column_stack([v] + u_copies)

                    # Calculate likelihood for empty neighborhood
                    empty_summary = m2sdt.associate_cdate(data, 0, [])
                    empty_summary = m2sdt.calculate_likelihood(empty_summary)
                    empty_summary = m2sdt.penalize_llhood(empty_summary, c, n_samples)

                    # Calculate likelihood for neighborhood
                    candidate = list(range(1, neighbor_size + 1))
                    candidate_summary = m2sdt.associate_cdate(data, 0, candidate)
                    candidate_summary = m2sdt.calculate_likelihood(candidate_summary)
                    candidate_summary = m2sdt.penalize_llhood(candidate_summary, c, n_samples)

                    # Determine if candidate neighborhood is better than empty
                    if candidate_summary['pen_llhood'] > empty_summary['pen_llhood']:
                        correct_count += 1

                # Store proportion of correct identifications
                snr = m3sdt.calculate_snr(p)
                results[neighbor_size][c].append((snr, correct_count / n_repeats))

    return results

def test_method(prob, c_cte, ne_num, rep_num, samp_size):
    """
    Tests the neighborhood detection method with balanced true/false cases.

    This function evaluates the performance of the penalized likelihood method by generating balanced
    true and false cases. For true cases, it creates a node and its noisy copies as the true neighborhood.
    For false cases, it creates a node and random nodes as the candidate neighborhood. It then compares
    the likelihood of the empty neighborhood versus the candidate neighborhood and calculates the true
    positive rate (TPR) and false positive rate (FPR) for detecting the neighborhood.

    Parameters:
    -----------
    prob : float
        Probability for flipping bits (noise level).
    c_cte : float
        Penalty constant for the likelihood.
    ne_num : int
        Number of nodes in the candidate neighborhood.
    rep_num : int
        Number of repetitions (must be even to balance true/false cases).
    samp_size : int
        Number of samples (realizations) to generate.

    Returns:
    --------
    SNR : float
        Signal-to-noise ratio, calculated as (1 - prob) / prob.
    TPR : float
        True positive rate (proportion of true neighborhoods correctly identified).
    FPR : float
        False positive rate (proportion of false neighborhoods incorrectly identified).

    Author: AI generated with orientation of Paulo Roberto Cabral Passos
    Date: October 20, 2025
    """
    # Balanced labels
    labels = np.array([0] * (rep_num // 2) + [1] * (rep_num // 2))
    np.random.shuffle(labels)
    results = np.zeros(labels.shape[0])

    for rep in range(labels.shape[0]):
        if labels[rep] == 1:
            # True neighbors: node 0 + ne_num-1 noisy copies
            v = np.random.randint(0, 2, samp_size)
            u = [m3sdt.flip_with_probability(v, prob) for _ in range(ne_num)]
            input_array = np.column_stack([v] + u)
        else:
            # All random nodes (node 0 + ne_num-1 irrelevant nodes + extra irrelevant nodes)
            v = np.random.randint(0, 2, samp_size)
            u = [np.random.randint(0, 2, samp_size) for _ in range(ne_num)]
            input_array = np.column_stack([v] + u)

        # Compare empty vs candidate neighborhood
        cdate = []
        empty_summary = m2sdt.associate_cdate(input_array, 0, cdate)
        empty_summary = m2sdt.calculate_likelihood(empty_summary)
        empty_summary = m2sdt.penalize_llhood(empty_summary, c_cte, samp_size)

        cdate = list(range(1, ne_num+1))
        cdate_summary = m2sdt.associate_cdate(input_array, 0, cdate)
        cdate_summary = m2sdt.calculate_likelihood(cdate_summary)
        cdate_summary = m2sdt.penalize_llhood(cdate_summary, c_cte, samp_size)

        if cdate_summary['pen_llhood'] > empty_summary['pen_llhood']:
            results[rep] = 1

    # Count TP/FP/TN/FN
    TP_count = np.sum((labels == 1) & (results == 1))
    FN_count = np.sum((labels == 1) & (results == 0))
    FP_count = np.sum((labels == 0) & (results == 1))
    TN_count = np.sum((labels == 0) & (results == 0))

    TPR = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0
    FPR = FP_count / (FP_count + TN_count) if (FP_count + TN_count) > 0 else 0
    SNR = (1 - prob) / prob
    return SNR, TPR, FPR

#TARGET FUNCTION
def generate_prediction_v2(noi_key, state, global_summaries):
    """
    """

    if noi_key not in global_summaries.keys():
        print('noi key must be in global_summaries')
        return 'no_output', 'no_output'
    
    associate_summary = global_summaries[noi_key]
    check_board = associate_summary['joint_configs']

    # Find all rows where the neighbor states match

    options = []
    for i in range(check_board.shape[0]):
        if check_board[i, 1:].tolist() == state:
            options.append(i)

    if len(options) > 0:
        # Find the most probable state of noi for this neighbor state
        best_idx = options[0]
        best_prob = associate_summary['probabilities'][best_idx]

        for idx in options[1:]:
            if associate_summary['probabilities'][idx] > best_prob:
                best_idx = idx
                best_prob = associate_summary['probabilities'][idx]

        prediction = associate_summary['joint_configs'][best_idx, 0]
        
    else:
        # Fallback: Use marginal probability of noi
        p0_num = 0
        p1_num = 0
        p_den = 0

        for i in range(check_board.shape[0]):
            if check_board[i, 0] == 0:
                p0_num += associate_summary['joint_count'][i]
            else:
                p1_num += associate_summary['joint_count'][i]
            p_den += associate_summary['joint_count'][i]

        p0_marg = p0_num / p_den
        p1_marg = p1_num / p_den

        prediction = 0 if p0_marg > p1_marg else 1
        if prediction == 0:
            best_prob = p0_marg
        else:
            best_prob = p1_marg

    return prediction, best_prob

def generate_prediction_v3(noi_key, state, global_summaries):
    """
    Returns the probability that the target choice is 1, given the neighbor states.
    """
    if noi_key not in global_summaries.keys():
        print('noi_key must be in global_summaries')
        return 0.0  # Default to 0 if the key is missing

    associate_summary = global_summaries[noi_key]
    check_board = associate_summary['joint_configs']

    # Find all rows where the neighbor states match
    options = []
    for i in range(check_board.shape[0]):
        if len(state) == len(associate_summary['cdate']):
            if check_board[i, 1:].tolist() == state:
                options.append(i)

    if len(options) > 0:
        # Find the probability that the target choice is 1 for this neighbor state
        for idx in options:
            if check_board[idx, 0] == 1:  # If the target is 1 in this configuration
                return associate_summary['probabilities'][idx]
        # If no matching configuration has target=1, return 0
        return 0.0
    else:
        # Fallback: Use marginal probability of the target being 1
        p1_num = 0
        p_den = 0
        for i in range(check_board.shape[0]):
            if check_board[i, 0] == 1:
                p1_num += associate_summary['joint_count'][i]
            p_den += associate_summary['joint_count'][i]
        return p1_num / p_den if p_den > 0 else 0.0  # P(target=1)



# TARGET FUNCTION
def predict_testdata(test_data, neighborhoods, global_summaries):
    """
    Generates predicted probabilities for each choice being 1 in the test set.
    Returns the predicted choice (0, 1, or 2) for each trial.
    """
    # Ensure test data have the correct indexes
    test_data = test_data.reset_index(drop=True)

    # Extract EEG features from the test set
    eeg_columns = [col for col in test_data.columns if not col.startswith('choice')]
    eeg_test = test_data[eeg_columns].values.astype(np.float32)

    # Initialize lists to store predictions and true labels
    predictions = []  # Predicted choice (0, 1, or 2) for each trial
    true_labels = []   # True choice (0, 1, or 2) for each trial

    # Loop over each trial in the test set
    for trial in range(eeg_test.shape[0]):
        choice_probs = []

        # Loop over each choice
        for choice in ['choice0', 'choice1', 'choice2']:
            associate_summary = global_summaries[choice]
            ne_indexes = [int(i) for i in associate_summary['cdate']]
            state = test_data.iloc[trial, ne_indexes].to_list() if ne_indexes else []

            # Get P(choice == 1 | neighbors)
            prob = generate_prediction_v3(choice, state, global_summaries)
            choice_probs.append(prob)

        # Normalize probabilities to sum to 1 (softmax-like)
        prob_sum = sum(choice_probs)
        if prob_sum > 0:
            normalized_probs = [p / prob_sum for p in choice_probs]
        else:
            # Fallback: uniform distribution if all probabilities are 0
            normalized_probs = [1/3, 1/3, 1/3]

        # Predict the choice with the highest probability
        predicted_choice = np.argmax(normalized_probs)
        predictions.append(predicted_choice)

        # Determine the true choice (0, 1, or 2)
        true_choice = -1
        for ilab, choice_lab in enumerate(['choice0', 'choice1', 'choice2']):
            if test_data[choice_lab].iloc[trial] == 1:
                true_choice = ilab
                break
        true_labels.append(true_choice)

    return predictions, true_labels

# bup > 
# def predict_testdata(test_data, neighborhoods, global_summaries):
#     """
#     Generates predicted probabilities for each choice in the test set using generate_prediction_v2.

#     Parameters:
#     -----------
#     test_data : pandas.DataFrame
#         The test set.
#     neighborhoods : dict
#         Estimated neighborhood matrices for each choice.
#     global_summaries : dict
#         Global summaries for each choice.

#     Returns:
#     --------
#     predictions : dict
#         A dictionary where keys are choice labels and values are tuples of (predicted label, probability) for each trial.
#     true_labels : dict
#         A dictionary where keys are choice labels and values are true labels for each trial.
#     """
#     # Ensuring test data have the correct indexes
#     test_data = test_data.reset_index(drop=True)
    
#     # Extract EEG features from the test set
#     eeg_columns = [col for col in test_data.columns if not col.startswith('choice')]
#     eeg_test = test_data[eeg_columns].values.astype(np.float32)
    
#     # Initialize dictionaries to store predictions and true labels
#     predictions = {choice: [] for choice in ['choice0', 'choice1', 'choice2']}
#     true_labels = {choice: [] for choice in ['choice0', 'choice1', 'choice2']}
    
#     # Loop over each trial in the test set
#     for trial in range(eeg_test.shape[0]):
#         # Extract the EEG features for the current trial
#         trial_features = eeg_test[trial, :]
    
#         # Loop over each choice
#         for choice in ['choice0', 'choice1', 'choice2']:
#             associate_summary = global_summaries[choice]
    
#             # Get the neighborhood state for the current trial
#             ne_indexes = [int(i) for i in associate_summary['cdate'] ]
#             if len(ne_indexes) == 0:
#                 state = list()
#             else:
#                 state = test_data.iloc[trial, ne_indexes].to_list()
    
#             # Generate prediction for the current choice using generate_prediction_v2
#             prediction, best_prob = generate_prediction_v2(choice, state, global_summaries)
    
#             predictions[choice].append((prediction, best_prob))
    
#             # Store the true label
#             true_labels[choice].append(test_data[choice].iloc[trial])

#     return predictions, true_labels
# bup <

def compute_auc(true_labels, predictions):
    """
    Computes the AUC for each choice based on predicted probabilities and true labels.

    Parameters:
    -----------
    true_labels : dict
        True labels for each choice.
    predictions : dict
        Predicted labels and probabilities for each choice.

    Returns:
    --------
    auc_scores : dict
        A dictionary where keys are choice labels and values are AUC scores.
    """
    auc_scores = {}

    for choice in ['choice0', 'choice1', 'choice2']:
        # Extract true labels and predicted probabilities
        y_true = true_labels[choice]
        y_probs = [prob for (_, prob) in predictions[choice]]

        # Compute AUC
        auc = roc_auc_score(y_true, y_probs)
        auc_scores[choice] = auc

    return auc_scores

# Check

def predict_final_label(predictions):
    """
    Determines the final predicted label for each trial based on the greatest probability.
    If all predicted labels are 0, the choice with the smallest probability is selected.

    Parameters:
    -----------
    predictions : dict
        Predicted labels and probabilities for each choice.
        Format: {'choice0': [(label, prob), ...], 'choice1': [(label, prob), ...], ...}

    Returns:
    --------
    final_labels : list
        A list of final predicted labels for each trial (0, 1, or 2).
    """
    final_labels = []
    n_trials = len(predictions['choice0'])
    choices = ['choice0', 'choice1', 'choice2']

    for trial in range(n_trials):
        pred_vec = []
        prob_vec = []
        for choice in choices:
            pred_vec.append(int(predictions[choice][trial][0]))  # Predicted label (0 or 1)
            prob_vec.append(float(predictions[choice][trial][1]))  # Probability (as float)

        # Case 1: Only one choice has pred_vec == 1
        if sum(pred_vec) == 1:
            final_label = [idx for idx, item in enumerate(pred_vec) if item == 1][0]

        # Case 2: Multiple choices have pred_vec == 1
        elif sum(pred_vec) > 1:
            search_list = [idx for idx, item in enumerate(pred_vec) if item == 1]
            final_label = search_list[0]
            for idx in search_list:
                if prob_vec[idx] > prob_vec[final_label]:
                    final_label = idx

        # Case 3: No choice has pred_vec == 1
        else:
            search_list = range(len(choices))  # All choices
            final_label = search_list[0]
            for idx in search_list:
                if prob_vec[idx] < prob_vec[final_label]:  # Pick the smallest probability
                    final_label = idx

        final_labels.append(final_label)

    return final_labels
