# Necessary packages

import scipy.io
import pandas as pd
import pickle
import numpy as np
import scipy.fft as spf

# Functions

def rd_from_matlab(data_folder, data_file, entry):
    """
    Description: 
        returns a tuple with the EEG data, goalkeeper game 
        matrix and recording info
    Input:
        data_folder = folder with the 'repoX.mat' file
        data_file = data file in the following format 'repoX.mat' where
        X is an integer
        entry = an integer that returns the corresponding entry of the
        repoX.mat. If the entry is superior to the number of subjects in
        the file, the function does not return any information
    Output:
        summary_tuple = (eeg_data, behav_df, recinfo_df) where 
        eeg_data is a numpy array, behav_df is a dataframe with goalkeeper
        game information and recinfo_df is a dataframe with additional info
    Author:
        Paulo Roberto Cabral Passos
    Last modified:
        30/04/2025
    """
    # Loading data
    output_data = scipy.io.loadmat(data_folder + '/' + data_file)

    # Checking if the entry exist
    if entry >= output_data['repo'].shape[0]:
        print('The entry exceeds the file content')
        return -1
    
    # Umpacking data 
    eeg_data = output_data['repo'][entry][0]
    behav_data = output_data['repo'][entry][1]
    fdt_file = output_data['repo'][entry][2][0][0][0][0]
    raw_file = output_data['repo'][entry][2][0][0][1][0].split(' ')[-1]
    srate = output_data['repo'][entry][2][0][0][2][0][0]
    mount_info = output_data['repo'][entry][2][0][0][3][0]
        
    # Reading channel names
    chan_names = list()
    for i in mount_info:
        chan_names.append( str(i[0][0]) + ' ' + str(mount_info[0][11][0]))
    
    # reading channel x coordinate
    chan_xcoor = list()
    for i in mount_info:
        if len(i[4]) == 0:
            chan_xcoor.append(0)
        else:
            chan_xcoor.append(float(i[4][0]))
    
    # reading channel y coordinate
    chan_ycoor = list()
    for i in mount_info:
        if len(i[5]) == 0:
            chan_ycoor.append(0)
        else:
            chan_ycoor.append(float(i[5][0]))
    
    # reading channel z coordinate
    chan_zcoor = list()
    for i in mount_info:
        if len(i[6]) == 0:
            chan_zcoor.append(0)
        else:
            if len(i[6][0]) != 0:
                chan_zcoor.append( float(i[6][0]) ) 
            else:
                chan_zcoor.append( float(0) ) 
    
    # converting to dataframes
    mount_dict = {'chan_name': chan_names, 'x_coor': chan_xcoor, 'y_coor': chan_ycoor, 'z_coor': chan_zcoor }
    mount_df = pd.DataFrame.from_dict(mount_dict)
    recinfo_dict = {'fdt': [fdt_file], 'raw': [raw_file], 'srate': [srate]}
    recinfo_df = pd.DataFrame.from_dict(recinfo_dict)
    behav_dict= {'trial_num': behav_data[:,2], 'tree': behav_data[:,4], 'id': behav_data[:,5]
                ,'rt': behav_data[:,6], 'gk_ch': behav_data[:,7], 'taker_ch': behav_data[:,8]
                ,'gx_trig': behav_data[:,9], 'g1_trig': behav_data[:,10], 'd2_arrow_trig': behav_data[:,11]
                ,'d3_fbini_trig': behav_data[:,12], 'd4_fbend_trig': behav_data[:,13], 'oid': behav_data[:,14]}
    behav_df = pd.DataFrame.from_dict(behav_dict)
    
    for c in behav_df.columns:
        if not c == 'rt':
            behav_df[c] = behav_df[c].astype('int')

    summary_tuple = (eeg_data,behav_df,recinfo_df, mount_df)
    return summary_tuple

def save_summ_to_file(file_name, summary_tuple):
    """
    Description: 
        save the summary_tuple (but can be any python data) to a file in
        the current folder with the name file_name
    Input:
        file_name = name for saving the data
        summary_tuple = data coming from rd_from_matlab
    Output:
        no output
    Author:
        Paulo Roberto Cabral Passos
    Last modified:
        05/05/2025
    """
    with open(file_name,'wb') as file:
        pickle.dump(summary_tuple)

def ss_spectrum(y,srate):
    """
    Description: 
        Returns the single-sided spectra from y after applying a Hanning window.
    Input:
        y = numpy array with the signal
        srate = sampling rate of the signal
    Output:
        ss_Y = single-sided spectrum of the signal. Note: in case of odd
        length signals, the last sample is discarted.
    Author:
        Paulo Roberto Cabral Passos
    Last modified:
        05/05/2025
    """
    # For function functionality test
    # ex.: Generating a 10Hz sinusoid
    #
    # srate = 256
    # x=np.array(range(srate))
    # w_0= 10
    # y=np.sin( x*(1/srate)*w_0*2*np.pi )
    
    # Checking for oddity
    y = y[0:-1] if y.shape[0]%2 != 0 else y
        
    
    t= np.array( range( y.shape[0]))/srate
    
    # Applying a Hanning window
    window = np.hanning(y.shape[0])
    y = y * window
    
    # Getting the single-sided representation
    Y=spf.fft(y, norm='forward')
    ss_Y = 2*np.abs( Y[0: int( Y.shape[0]/2 ) ] )
    ss_k_l = int( Y.shape[0]/2 )
    ss_k = np.array( range(ss_k_l) )*(1/ss_k_l)*ss_k_l

    return (ss_Y, ss_k)

def get_loc_plotting(num_plots, lnumber, colnumber, n_plot):
    """
    Description (simplified): 
        return the indexes for plotting using plt.subplots given the number of plots <num_plots>, the number
        of lines <lnumber> and the number of columns <colnumber>. The indexes are returned for the <n_plot>    
    Author:
        Paulo Roberto Cabral Passos
        
    Last modified:
        03/06/2025
    """     
    grid = np.reshape(  np.array( range(num_plots) ), (lnumber,colnumber)  )
    return np.where(grid == n_plot)

def get_snippets(eeg_data, behavior_df, chan, marker, phrom, till, cut_size, orientation, jump):
    """
    Description: 
        return the requested signal snipts according to the provided parameters
        
    Input:
        eeg_data = a numpy array EEG recordings shaped (channels, time samples) 
        behavior_df = a data frame with goalkeeper game information. The data frame
                      shaped (trials, parameters)
        chan = integer indicating the channel from which the snipts will be extracted
        marker = the available markers are: 'gx_trig', 'g1_trig', 'd2_arrow_trig',
        'd3_fbini_trig', 'd4_fbend_trig'         
        phrom = from which trial the snipts will start to be pushed into the output array
        till = until which trial the snipts will be pushed into the output array
        cut_size = size of the snippet (in samples)
        orientation = the available orientations are 'backward' and 'forward'
        jump = a number of samples that will be shifted from the marker to pushing the 
        snippet according to the orientation. 0 provides no shift.
        
    Output:
        snippets_repo = numpy array shapped (trials, snippet)
        trial_ref = indicates from which trial each snippet was retrieved.
        
    Author:
        Paulo Roberto Cabral Passos
        
    Last modified:
        21/05/2025
    """
    
    snippets_repo = np.zeros( (till - phrom + 1, cut_size) )
    trial_ref = list()
    aux = 0
    for k in range(phrom-1,till):
        cut_from = behavior_df[marker][k]-1
        if orientation == 'backward':
            cut_from = cut_from - jump
            cut_till = cut_from - cut_size
        else:
            cut_from = cut_from + jump
            cut_till = cut_from + cut_size       
        try:
            x = eeg_data[chan, cut_from: cut_till] if orientation == 'forward' else eeg_data[chan, cut_till: cut_from]
            snippets_repo[aux,:] = x
            trial_ref.append(k)
            aux +=1
        except Exception as error_messege:
            print('Unable to process the requested for trial ', k)
            
    return snippets_repo, trial_ref

def get_spectra(snippets_repo, srate):
    spec_list = list()
    for row in range( snippets_repo.shape[0] ):
        spec, ssk = ss_spectrum( snippets_repo[row,:], srate )
        spec_list.append(spec)
    spec_repo = np.stack(spec_list)
    return spec_repo, ssk

def chan_desync_window_scan(trial_window_len, trial_shift_len, max_d, srate, sample_window,
                          alphalft, alpharght, ch, d_fix, fix_apply_to, R_orient, A_orient):
    """
    Description: 
        Return a array of desynchronization coefficients that can be used to choose the best resting window
        for the analysis.
        
    Input:
        trial_window_len = length in trials of a window that will be slided across the total number of trials
        trial_shift_len = length of trial slide across the total number of windows
        max_d = maximum displacement from the marker in samples
        srate = sampling rate of the recordings
        sample_window = length in samples of the resting window.
        alpha_lft = indicate from which frequency we should consider the "beggining of alpha"
        alpha_rght = indicate from which frequency we should consider the "end of alpha"
        eeg_data = array with the eeg recordings
        behavior_df = data frame with the goalkeeper information
        ch = channel to be processed
        
    Output:
        window_dist_curve = each row this array presents the coefficient of desynchronization as a function of
        the displacement.
        dist_list = indicate for each row of window_dist_curve, which displacement results in maximum desyn-
        cronization.
        displace_list = indicate for each row of window_dist_curve the respective displacement
        
    Author:
        Paulo Roberto Cabral Passos
        
    Last modified:
        22/05/2025
    """
    global eeg_data, behavior_df
    max_num =int( (behavior_df.shape[0] - trial_window_len)/trial_shift_len )
    output2_list = list()
    for t in range(max_num+1):
        trial_e = trial_window_len + t*trial_shift_len
        trial_b = trial_e - trial_window_len+1   
        displace_list = list()
        output1_list = list()
        for d in range(0,max_d):
            if fix_apply_to == 'A':
                snippets_repoR, trial_refR = get_snippets(eeg_data, behavior_df, ch, 'd4_fbend_trig', trial_b, trial_e, sample_window, R_orient, d_fix)
                snippets_repoA, trial_refA = get_snippets(eeg_data, behavior_df, ch, 'gx_trig', trial_b, trial_e, sample_window, A_orient, d)
            else:
                snippets_repoR, trial_refR = get_snippets(eeg_data, behavior_df, ch, 'd4_fbend_trig', trial_b, trial_e, sample_window, R_orient, d)
                snippets_repoA, trial_refA = get_snippets(eeg_data, behavior_df, ch, 'gx_trig', trial_b, trial_e, sample_window, A_orient, d_fix)
            spec_repoR , ssk = get_spectra( snippets_repoR, srate )
            spec_repoA , ssk = get_spectra( snippets_repoA, srate )
            int_liml = np.where(ssk == alphalft)[0].item()
            int_limr = (np.where(ssk == alpharght)[0]+1).item()
            coefficient = 0
            fy = np.subtract(spec_repoR, spec_repoA)
            for r in range(spec_repoR.shape[0]):
                coefficient += np.trapz( fy[r, int_liml:int_limr] )
            displace_list.append(d)
            output1_list.append(coefficient/spec_repoR.shape[0])
            output1_arr = np.array(output1_list)
        output2_list.append(output1_arr)
    
    window_dist_curve = np.stack(output2_list, axis=0)
    dist_list = list()
    for w in range( window_dist_curve.shape[0] ):
        dist_list.append( np.argmax(window_dist_curve[w, :]) )
    return (window_dist_curve, dist_list, displace_list)




