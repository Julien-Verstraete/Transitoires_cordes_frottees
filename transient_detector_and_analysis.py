"""
Created on Wed Mar 19 18:56:45 2025

@author: julien
"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy.io
import glob


display = 'OFF'
savefile = 'OFF'
csv_filename1 = 'analyses_transitoires_piezzo'
csv_filename2 = 'analyses_transitoires_pickup'
# csv_filename = 'm1_spf-semitone_std-06_Astring'
main_path = file_path = 'C:/Users/Julie/Downloads/project transitoires/'

string = 'A'  # 'A' or 'C'
string_frequency = {'A': 220, 'C': 65.4}

# Parameters for sliding window algorithms
hop_size = 128
win_len = 2048

# Peak-picking algorithm parameters pp_parameters
pp_prominences = {'A': 0.013, 'C': 0.013}
pp_distances = {'A': 20, 'C': 20}
pp_height_factor = 1/2


#listes pour les statistiques des transitoires:
moy_ecart_transitoires = []
std_ecart_transitoires = []

nbre_transi_piezzo = []
nbre_transi_pickup = []
duree_transitoire_piezzo = []
duree_transitoire_pickup = []
fin_transitoire_piezzo =  []
fin_transitoire_pickup = []

# Threshold for Sliding Phase Frequency (bandwidth centered at string natural
# frequency):
#   spf_th_low = string_frequency[string]/2**(1/spf_th_wide)
#   spf_th_up = string_frequency[string]*2**(1/spf_th_wide)
# Common values:
#  24 -> semitone bandwidth
#  12 -> tone bandwidth
spf_th_wide = 24

# Threshold for the maximum standard deviation of the sliding phases timestamps
std_th = 0.6


os.chdir(main_path)


def detect_onsets_from_peaks(abs_sig_diff, peaks, sr, lagtime_peak=0.1):
    lagtime_peak_samples = int(lagtime_peak*sr)
    onset_frames = np.array([])
    
    time = np.linspace(0,len(abs_sig_diff)/sr,len(abs_sig_diff)) # time vector of the analyzed signal
    seuil_d_erreur = 15 # duration (number of period) when we want to make sure there are peaks after the onset's detection 
    # Only a peak can be an onset
    for i in range(len(peaks)):
        # If the peaks are in the first lagtime_peak seconds, we can not know
        # if it is an onset
        if peaks[i] < lagtime_peak_samples:
            continue
        
        # Check if we have enough peaks remaining in the list
        if i + seuil_d_erreur >= len(peaks):
            break  # Sortir de la boucle si on dépasse la taille de peaks
            
            
        # Number of peaks detected lagtime_peak seconds before current peak
        arethey_peaks_before = len(np.where(
            np.logical_and(peaks >= peaks[i] - lagtime_peak_samples,
                           peaks < peaks[i]))[0])
        # If there are no peaks lagtime_peak seconds before current peak
        # --> ONSET DETECTED
        if arethey_peaks_before == 0:
                if (time[peaks[i+seuil_d_erreur]] - time[peaks[i]]) <= (1.5*seuil_d_erreur)/220: # we look if between the detected onset and the tolerance (seuil_d_erreur), there are other peaks, making sure that we are after a bow stroke    
                    onset_frames = np.append(onset_frames, peaks[i])  #Thus, the duration between the tolerance and the following peaks must be inferior than 1.5 * seuil_d_erreur (because some strokes have more inconsistent frequencies before Helmholtz motion)
                else:
                    None
        
    return onset_frames/sr


count = 0

# Datatype of the transient dataset FOR BOXPLOT
dtype = np.dtype([('Transient number', int, 1),
                  ('Bow part', np.unicode_, 512),
                  ('Dynamics', np.unicode_, 512),
                  ('Direction', np.unicode_, 512),
                  ('Musician', np.unicode_, 512),
                  ('Bow', np.unicode_, 512),
                  ('Transient start', int, 1),
                  ('Transient end', int, 1)])
# Create empty transient array that will contain all transients of all signals
all_transients_array_piezzo = np.ndarray(shape=1,  # first row to delete
                                  dtype=dtype)

all_transients_array_pickup = np.ndarray(shape=1,  # first row to delete
                                  dtype=dtype)
# --- Progress bar

# Count number of .mat files
nb_files = 0
for filename in glob.glob('*' + string + 'string.mat'):
    nb_files += 1
    
print("nombre de fichiers dans l'analyse:",nb_files)

# Create progress bar
pvar = tqdm(desc='Transient detection in progress... ', total=nb_files)


pp_height_factor_array = np.linspace(0.1,0.75,10)

# def transitoire_detecte(sig, all_transients_array,peaks,onset_frame,sr):
#     for i in range(len(onset_frame)):
#         if all_transients_array[onset_frame][6]
    
#     return None

# --- Main loop

for p in range(len(pp_height_factor_array)):

    # For every signal (.mat file) in main_path
    for filename in glob.glob('*' + string + 'string.mat'):
    
        # -- Load signal and sampling rate
        file_path = main_path + "/" + filename
        print('File: ' + filename)
        sr = int(scipy.io.loadmat(file_path)['freq'])
        sig = scipy.io.loadmat(file_path)['indata'][:, 0]
        pickup = scipy.io.loadmat(file_path)['indata'][:,2]
        # -- Calculate first difference of the signal in absolute value
        abs_sig_diff = np.abs(np.diff(sig))
        abs_pick_diff = np.abs(np.diff(pickup))
        
        # -- Detect peaks from abs_sig_diff, i.e., detect slipping phases
    
        # Peak-picking algorithm parameters (pp): see pp_parameters section
    
        # Minimum height of the peaks: Adaptative threshold varying with
        # abs_sig_diff envelope
        # --> Minimum height = pp_height = maximum peak of the windowed
        #                      abs_sig_diff signal times factor
    
        pp_height_piezzo = np.zeros_like(abs_sig_diff)
        pp_height_pickup = np.zeros_like(abs_pick_diff)
    
        for i in range(len(abs_sig_diff)-win_len):
            pp_height_piezzo[int(win_len/2)+i] = np.max(abs_sig_diff[i:win_len+i]) \
                * pp_height_factor_array[p]
    
        # Adjust initial and final height of signal (if not, height=0)
        pp_height_piezzo[0:int(win_len/2)] = pp_height_piezzo[int(win_len/2)]
        pp_height_piezzo[-int(win_len/2):] = pp_height_piezzo[-int(win_len/2)-1]
    
    
        for i in range(len(abs_sig_diff)-win_len):
            pp_height_pickup[int(win_len/2)+i] = np.max(abs_pick_diff[i:win_len+i]) \
                * pp_height_factor_array[p]
    
        # Adjust initial and final height of signal (if not, height=0)
        pp_height_pickup[0:int(win_len/2)] = pp_height_pickup[int(win_len/2)]
        pp_height_pickup[-int(win_len/2):] = pp_height_pickup[-int(win_len/2)-1]
    
        # Detect peaks
        peaks1, _ = scipy.signal.find_peaks(abs_sig_diff,height = pp_height_piezzo, prominence=(0.015*np.max(abs_sig_diff),np.max(abs_sig_diff)),
                                           distance=1)
        peaks2, _ = scipy.signal.find_peaks(abs_pick_diff,height = pp_height_pickup, prominence=(0.015*np.max(abs_pick_diff),np.max(abs_pick_diff)),
                                           distance=40)
        # -- Detect onsets from peaks (first peak after 0.1 seconds of no peaks)
        file_onsets1 = detect_onsets_from_peaks(abs_sig_diff, peaks1, sr,
                                               lagtime_peak=0.1)
        file_onsets2 = detect_onsets_from_peaks(abs_pick_diff, peaks2, sr,
                                               lagtime_peak=0.1)
        # -- Calculate number of peaks from abs_sig_diff signal, i.e. :
        # nb_peaks = number of slips in the temporal window of length = win_len
        nb_peaks1 = np.zeros_like(abs_sig_diff)
        nb_peaks2 = np.zeros_like(abs_pick_diff)
        for i in range(len(abs_sig_diff)-win_len):
            nb_peaks1[int(win_len/2)+i] = len(np.where(
                np.logical_and(peaks1 >= i, peaks1 <= (win_len+i)))[0])
            nb_peaks2[int(win_len/2)+i] = len(np.where(
                np.logical_and(peaks2 >= i, peaks2 <= (win_len+i)))[0])  
            
        # -- ONSETS LOOP
    
        # Transient number index (transients are saved with an index for every
        # signal, starting from 0)
        transient_number_piezzo = 0
        transient_number_pickup = 0
        
        # Array to store Slipping Phase (pseudo)Frequency (spf) values in Hz
        spf_array1 = np.zeros_like(abs_sig_diff)
        spf_array2 = np.zeros_like(abs_pick_diff)
        # Array to store standard deviation of the peak distribution in the
        # windowed abs_sig_diff signal
        std_peaks_array1 = np.zeros_like(abs_sig_diff)
        std_peaks_array2 = np.zeros_like(abs_pick_diff)
        # For every onset detected (and then transient candidate):
        for i in range(len(file_onsets1)):
    
            # -- Calculate transient length
            # How? -> parsing signal from current onset to next onset and
            # analysing mean and standard deviation
    
            # If last onset, then next onset = end of signal
            
            if i+1 == len(file_onsets1):
                next_onset = len(abs_sig_diff)
            else:
                next_onset = file_onsets1[i+1]
    
            # End of transient, set to -1 in case of no transient detected
            transient_end = -1
    
            # nb_peaks_cut = part of the current nb_peaks function (from current
            # onset to next onset or end of signal)
            nb_peaks_cut1 = nb_peaks1[int(file_onsets1[i]*sr):int(next_onset*sr)]
            
            # -- ANALYSIS OF THE FREQUENCY AND STD OF THE SLIPS TO DETECT HELMHOLTZ
            # Sliding window algorithm to analyse both spf_array and str_peaks
            # functions, both with thresholds to detect the beginning of the
            # Helmholtz motion and thus the end of the transient
            for j in range(len(nb_peaks_cut1)-win_len):
                first_sample = int(file_onsets1[i]*sr)+j
                last_sample = int(file_onsets1[i]*sr)+j+win_len
                # Mean number of peaks in the current windowed nb_peaks function
                mean_peaks1 = np.mean(nb_peaks1[first_sample:last_sample])
                # Slipping Phase (pseudo)Frequency (spf) in Hz
                
                spf1 = sr*mean_peaks1/win_len
                # Std
                std_peaks1 = np.std(nb_peaks1[first_sample:last_sample])
                
                # store both spf and std in respective arrays
                spf_array1[first_sample] = spf1
                std_peaks_array1[first_sample] = std_peaks1
    
                # - MAIN DETECTING CONDITION:
                # Helmholtz regime is reached if:
                #   - frequency = frequency(string) in some bandwidth threshold
                #                   (spf_th_wide)
                #   - std < std_th
                # THEN --> break loop and go to next detected onset
    
                # Lower and upper threshold for slipping phase frequency (spf)
                spf_th_low = string_frequency[string]/2**(1/spf_th_wide)
                spf_th_up = string_frequency[string]*2**(1/spf_th_wide)
                if spf_th_low < spf1 < spf_th_up:
                    if std_peaks1 < std_th:
                        transient_end = int(first_sample)  # in samples
                        break
    
            # If Helmholtz regime never reached, transient ommited (probably caused
            # by false onset)
            if transient_end == -1:
                continue
    
            transients_array_piezzo = np.ndarray(shape=1, dtype=dtype)
            transients_array_piezzo[0][0] = transient_number_piezzo
            transients_array_piezzo[0][1] = filename.split('_')[1]
            transients_array_piezzo[0][2] = filename.split('_')[2]
            transients_array_piezzo[0][3] = filename.split('_')[3]
            transients_array_piezzo[0][4] = filename.split('_')[4]
            transients_array_piezzo[0][5] = filename.split('_')[5]
            transients_array_piezzo[0][6] = int(file_onsets1[i]*sr)
            transients_array_piezzo[0][7] = transient_end
    
            # transient_lims = np.append(transient_lims,
            #                            np.array([filename, int(transient_number),
            #                                      int(file_onsets[i]*sr),
            #                                      transient_end]))
    
            # add transients detected to dataset
            all_transients_array_piezzo = np.append(all_transients_array_piezzo,
                                             transients_array_piezzo)
    
            transient_number_piezzo += 1
    
    
    
    
    
    
        for i in range(len(file_onsets2)):
    
            # -- Calculate transient length
            # How? -> parsing signal from current onset to next onset and
            # analysing mean and standard deviation
    
            # If last onset, then next onset = end of signal
            if i+1 == len(file_onsets2):
                next_onset = len(abs_pick_diff)
            else:
                next_onset = file_onsets2[i+1]
    
            # End of transient, set to -1 in case of no transient detected
            transient_end = -1
    
            # nb_peaks_cut = part of the current nb_peaks function (from current
            # onset to next onset or end of signal)
            nb_peaks_cut2 = nb_peaks2[int(file_onsets2[i]*sr):int(next_onset*sr)]
            
            # -- ANALYSIS OF THE FREQUENCY AND STD OF THE SLIPS TO DETECT HELMHOLTZ
            # Sliding window algorithm to analyse both spf_array and str_peaks
            # functions, both with thresholds to detect the beginning of the
            # Helmholtz motion and thus the end of the transient
            for j in range(len(nb_peaks_cut2)-win_len):
                first_sample = int(file_onsets2[i]*sr)+j
                last_sample = int(file_onsets2[i]*sr)+j+win_len
                # Mean number of peaks in the current windowed nb_peaks function
                mean_peaks2 = np.mean(nb_peaks2[first_sample:last_sample])
                # Slipping Phase (pseudo)Frequency (spf) in Hz
                
                spf2 = sr*mean_peaks2/win_len
                # Std
                std_peaks2 = np.std(nb_peaks2[first_sample:last_sample])
                
                # store both spf and std in respective arrays
    
                spf_array2[first_sample] = spf2
                std_peaks_array2[first_sample] = std_peaks2
                
                # - MAIN DETECTING CONDITION:
                # Helmholtz regime is reached if:
                #   - frequency = frequency(string) in some bandwidth threshold
                #                   (spf_th_wide)
                #   - std < std_th
                # THEN --> break loop and go to next detected onset
    
                # Lower and upper threshold for slipping phase frequency (spf)
                spf_th_low = string_frequency[string]/2**(1/spf_th_wide)
                spf_th_up = string_frequency[string]*2**(1/spf_th_wide)
                if spf_th_low < spf2 < spf_th_up:
                    if std_peaks2 < std_th:
                        transient_end = int(first_sample)  # in samples
                        break
    
            # If Helmholtz regime never reached, transient ommited (probably caused
            # by false onset)
            
            
            if transient_end == -1:
                continue
    
            transients_array_pickup = np.ndarray(shape=1, dtype=dtype)
            transients_array_pickup[0][0] = transient_number_pickup
            transients_array_pickup[0][1] = filename.split('_')[1]
            transients_array_pickup[0][2] = filename.split('_')[2]
            transients_array_pickup[0][3] = filename.split('_')[3]
            transients_array_pickup[0][4] = filename.split('_')[4]
            transients_array_pickup[0][5] = filename.split('_')[5]
            transients_array_pickup[0][6] = int(file_onsets2[i]*sr)
            transients_array_pickup[0][7] = transient_end
    
            # transient_lims = np.append(transient_lims,
            #                            np.array([filename, int(transient_number),
            #                                      int(file_onsets[i]*sr),
            #                                      transient_end]))
    
            # add transients detected to dataset
            all_transients_array_pickup = np.append(all_transients_array_pickup,
                                             transients_array_pickup)
    
            transient_number_pickup += 1
            
            
        #ici on traite la durée des transitoires
    
    
        time = np.linspace(0,len(abs_sig_diff)/sr,len(abs_sig_diff))
        
    
        
        
        if len(file_onsets1) != len(file_onsets2):
            print("onsets numbers don't match")
    
                
        else:
            for j in range(len(std_peaks_array1) - 1):
                if (std_peaks_array1[j] !=0 and std_peaks_array1[j+1] == 0) :
                    fin_transitoire_piezzo.append(time[j])
                if (std_peaks_array2[j] !=0 and std_peaks_array2[j+1] == 0) :
                    fin_transitoire_pickup.append(time[j])
                        
            for i in range(len(file_onsets1)):
                duree_transitoire_piezzo.append(fin_transitoire_piezzo[i] - file_onsets1[i])
                duree_transitoire_pickup.append(fin_transitoire_pickup[i] - file_onsets2[i])
                    
                    
        if duree_transitoire_piezzo and duree_transitoire_pickup:
            duree_transitoire_piezzo = np.array(duree_transitoire_piezzo)
            duree_transitoire_pickup = np.array(duree_transitoire_pickup)
            moy_ecart_transitoires.append(np.mean(np.abs(duree_transitoire_piezzo - duree_transitoire_pickup)))
            std_ecart_transitoires.append(np.std(duree_transitoire_piezzo - duree_transitoire_pickup))
    
        #on réinitialise les listes:
            
        duree_transitoire_piezzo = []
        duree_transitoire_pickup = []
        fin_transitoire_piezzo =  []
        fin_transitoire_pickup = []
        
        if display == 'ON':
            fig, ax = plt.subplots(nrows=5, sharex=True)
            ax[0].plot(np.arange(len(sig))/sr, sig,color = 'blue', label='Signal Piezzo')
            ax[0].set(title=filename)
            ax[1].plot(np.arange(len(abs_sig_diff))/sr, abs_sig_diff, color = 'blue',
                        label='First derivative signal (absolute value) Piezzo')
            ax[1].scatter((np.arange(len(np.diff(sig)))/sr)[peaks1],
                          np.abs(np.diff(sig))[peaks1], label='Detected peaks from Piezzo Signal',
                          c='purple', marker='*')
            ax[1].plot(np.arange(len(pp_height_piezzo))/sr, pp_height_piezzo, color='yellow',
                        label='Maximum height for peak detection for piezzo')
            ax[1].plot(np.arange(len(pp_height_pickup))/sr, pp_height_pickup, color='purple',
                        label='Maximum height for peak detection for pickup')
            ax[2].plot(np.arange(len(nb_peaks1))/sr, nb_peaks1, color='blue',
                        label='Number of peaks (nb_peaks) from Piezzo')
            ax[0].vlines(file_onsets1, -5, 5, color='r', alpha=0.9,
                          linestyle='--', label='Detected onsets')
            ax[1].vlines(file_onsets1, 0, 0.6, color='r', alpha=0.9,
                          linestyle='--')
            ax[2].vlines(file_onsets1, 0, 40, color='r', alpha=0.9,
                          linestyle='--')
            ax[0].plot(np.arange(len(pickup))/sr, pickup, color = 'orange', label='Signal Pickup')
    
            ax[1].plot(np.arange(len(abs_pick_diff))/sr, abs_pick_diff, color = 'orange',
                        label='First derivative signal (absolute value) pickup')
            ax[1].scatter((np.arange(len(np.diff(pickup)))/sr)[peaks2],
                          np.abs(np.diff(pickup))[peaks2], color = 'red',label='Detected peaks from Pickup Signal', marker='*')
            ax[2].plot(np.arange(len(nb_peaks2))/sr, nb_peaks2,color = 'orange',
                        label='Number of peaks (nb_peaks) pickup')
    
            
            ax[0].legend(frameon=False)
            ax[1].legend(frameon=False)
            ax[2].legend(frameon=False)
    
            for k in range(transient_number_piezzo):
                ax[0].axvspan(int(all_transients_array_piezzo[-k-1][6])/sr,
                              int(all_transients_array_piezzo[-k-1][7])/sr, facecolor='g',
                              alpha=0.5)
                
            for k in range(transient_number_pickup):
                ax[0].axvspan(int(all_transients_array_pickup[-k-1][6])/sr,
                              int(all_transients_array_pickup[-k-1][7])/sr, facecolor='darkgreen',
                              alpha=0.5)
                
            ax[3].plot(np.arange(len(spf_array1))/sr, spf_array1, color='blue',
                        label='Slipping phase frequency (SPF) Piezzo')
            ax[3].plot(np.arange(len(spf_array2))/sr, spf_array2, color='orange',
                        label='Slipping phase frequency (SPF) pickup')
            ax[4].plot(np.arange(len(std_peaks_array2))/sr, std_peaks_array2, color = 'orange',label='Standard deviation (STD) pickup')
            ax[3].hlines(string_frequency[string]/2**(1/24), 0, 10, color='g',
                          alpha=0.9, linestyle='--',
                          label='Upper and lower SPF thresholds')
            ax[3].hlines(string_frequency[string]*2**(1/24), 0, 10, color='g',
                          alpha=0.9, linestyle='--')
            ax[3].legend(frameon=False)
            ax[4].plot(np.arange(len(std_peaks_array1))/sr, std_peaks_array1,
                        color='blue', label='Standard deviation (STD) Piezzo')
            ax[4].hlines(0.6, 0, 10, color='g', alpha=0.9,
                          linestyle='--', label='Upper STD threshold')
            ax[4].legend(frameon=False)
            ax[3].vlines(file_onsets1, 0, 440, color='r', alpha=0.9,
                          linestyle='--')
            ax[4].vlines(file_onsets1, 0, 10, color='r', alpha=0.9,
                          linestyle='--')
            ax[4].set_xlabel('Time (s)')
            ax[4].set_ylabel('STD (Slips)')
            ax[3].set_ylabel('SPF (Hz)')
            ax[2].set_ylabel('Slips')
            ax[1].set_ylabel('Amplitude')
            ax[0].set_ylabel('Amplitude')
            plt.show()
            
    
    
    
        # For testing just a bunch of signals
        # if count == 3:
        #     break
        
        count += 1
        pvar.update(1)

    nbre_transi_pickup.append(len(all_transients_array_pickup))
    nbre_transi_piezzo.append(len(all_transients_array_piezzo))
    
pvar.close()

print(".")
#print("ecart moyen de durée de transitoire par fichier",moy_ecart_transitoires)
print("écart moyen total en durée:",np.mean(moy_ecart_transitoires))
#print("variance sur la durée des transitoires par fichier:",std_ecart_transitoires)
print("variance totale sur la durée des transitoires",np.mean(std_ecart_transitoires))


# delete empty first row
all_transients_array_piezzo = np.delete(all_transients_array_piezzo, 0, 0)

all_transients_array_pickup = np.delete(all_transients_array_pickup, 0, 0)
# --- SAVE FILE WITH DETECTED TRANSIENTS FOR BOXPLOT



if savefile == 'ON':
    np.savetxt(csv_filename1 + '.csv', all_transients_array_piezzo, delimiter=';',
               fmt='%s')
    np.savetxt(csv_filename2 + '.csv', all_transients_array_pickup, delimiter=';',
               fmt='%s')
    
    
#on regarde si le nombre de périodes transitoires détectées entre le piezzo et le pickup matchent


count_pickup = 0
count_piezzo = 0
for i in range(len(nbre_transi_pickup)):
    count_pickup += nbre_transi_pickup[i]
    
for j in range(len(nbre_transi_piezzo)):
    count_piezzo += nbre_transi_piezzo[j]
    
print("count piezzo:",count_piezzo)
print("count pickup",count_pickup)

plt.figure()
plt.plot(pp_height_factor_array, nbre_transi_piezzo,label="piezzo")
plt.plot(pp_height_factor_array,nbre_transi_pickup,label="pickup")
plt.xlabel("pp_height_factor")
plt.ylabel("nombre de transitoires efficacement détectés")
plt.legend()
plt.grid()
plt.show()
