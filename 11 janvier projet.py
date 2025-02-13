# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:30:13 2025

@author: julien
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:19:34 2025

@author: julien
"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy.io
import glob
import scipy.signal as sg

file = scipy.io.loadmat("DVC_pointe_piano_pousse_musician1_BowB_Astring.mat")
indata = file['indata']

fe = 51200
pickup = indata
N = len(indata)
t = np.arange(0,N/fe,1/fe)



##############################################•

# fig, ax  = plt.subplots(1,2,figsize=(8,6), sharex=True)


# ax[0].plot(t,indata[:,0],label = "piezzo")
# ax[0].set_ylabel("amplitude")
# ax[0].legend()
# ax[0].grid()


# ax[1].plot(t,indata[:,2],label = "pickup")
# ax[1].set_ylabel("amplitude")
# ax[1].legend()
# ax[1].grid()

# plt.show()

############################################################
# faire un code qui détecte les transitoires sur le signal renvoyé par le pickup
# comparer les résultats entre le pickup et le piezzo

# plt.figure()
# diff = np.abs(indata[:,0]-indata[:,2])

# plt.plot(t,diff)
# plt.show()


###################################
#initialisation du code de victor
#définition de x' et find_peaks

x = []
xprime = []
for i in range(N):
    x.append(indata[i,0])
    
for i in range(N-1):
    xprime.append(x[i+1]-x[i])
    
    
win_len = 2048
pp_height_factor = 1/3
pp_prominences = 0.013
pp_distance = 20


xprime = np.array(np.abs(xprime))

pp_height = np.zeros_like(xprime)

for i in range(len(xprime)-win_len):
    pp_height[int(win_len/2)+i] = np.max(xprime[i:win_len+i]) \
        * pp_height_factor

# Adjust initial and final height of signal (if not, height=0)
pp_height[0:int(win_len/2)] = pp_height[int(win_len/2)]
pp_height[-int(win_len/2):] = pp_height[-int(win_len/2)-1]


peaks1 ,_ = sg.find_peaks(xprime,height = pp_height,prominence = pp_prominences)

# plt.figure()
# plt.plot(t[:511999],xprime)
# plt.plot(t[peaks1],xprime[peaks1],'x')

# plt.show()


##########~~~~~~~~~~~~~~~~~~~~##########################
#la méthode fonctionne, on essaie donc de l'appliquer sur le signal du pickup

xpick = []
xpick_diff = []
for i in range(N):
    xpick.append(indata[i,2])
    
for i in range(N-1):
    xpick_diff.append(xpick[i+1]-xpick[i])
    
xpick_diff = np.array(np.abs(xpick_diff))

peaks2,_ = sg.find_peaks(xpick_diff,height = pp_height,distance = 100,prominence = pp_prominences)

# plt.figure()
# plt.plot(t[:511999],xpick_diff)
# plt.plot(t[peaks2],xpick_diff[peaks2],'x')

##################################################################################"
#Analyse du délai entre les signaux pickup et piezzo
#étude statistique

data_pick= [0.06279,1.13189,2.168397,3.202477,4.23484,5.25185,6.297388]
data_piezzo = [0.06188,1.13112,2.1675,3.20168,4.23422,5.25115,6.296646]

d = []
for i in range(len(data_pick)):
    d.append(data_pick[i]-data_piezzo[i])
    
moy = np.mean(d)
# print("moyenne diff:",moy)


#################################################################################"
#on regarde à présent si le début du transitoire du pickup correspond au début du transitoire piezzo
#auquel on a rajouté le délai :
    
d2 = []
ppiezzo = [1.1458,2.1824,3.2167,4.2346,5.2609,6.3021]
ppickup = [1.1506,2.1825,3.2168,4.2347,5.2656,6.3023]

for i in range(len(ppiezzo)):
    d2.append(ppickup[i]-ppiezzo[i])
    
moy2 = np.mean(d2)

print("diff2:", moy2)

#######################################################################

spf_array = np.zeros_like(xprime)


sig = indata[:,2]

abs_sig_diff = np.abs(np.diff(sig))
std_peaks_array = np.zeros_like(abs_sig_diff)


nb_peaks = np.zeros_like(abs_sig_diff)
for i in range(len(abs_sig_diff)-win_len):
    nb_peaks[int(win_len/2)+i] = len(np.where(
        np.logical_and(peaks2 >= i, peaks2 <= (win_len+i)))[0])
    
# plt.figure()
# plt.plot(np.arange(len(nb_peaks))/fe,nb_peaks,label = "signal pickup")


sig2 = indata[:,0]

abs_sig_diff2 = np.abs(np.diff(sig2))
std_peaks_array = np.zeros_like(abs_sig_diff2)

nb_peaks2 = np.zeros_like(abs_sig_diff2)
for i in range(len(abs_sig_diff2) - win_len):
    nb_peaks2[int(win_len/2)+i] = len(np.where(np.logical_and(peaks1>=i,peaks1<=(win_len+i)))[0])
    
# plt.plot(np.arange(len(nb_peaks2))/fe,nb_peaks2,label = "signal piezzo")
# plt.legend()
# plt.grid()
# plt.show()

#############################################################################################

spf_th_wide = 24
std_th = 0.6

def detect_onsets_from_peaks(abs_sig_diff, peaks, sr, lagtime_peak=0.1):
    lagtime_peak_samples = int(lagtime_peak*sr)
    onset_frames = np.array([])
    # Only a peak can be an onset
    for i in range(len(peaks)):
        # If the peaks are in the first lagtime_peak seconds, we can not know
        # if it is an onset
        if peaks[i] < lagtime_peak_samples:
            continue
        # Number of peaks detected lagtime_peak seconds before current peak
        arethey_peaks_before = len(np.where(
            np.logical_and(peaks >= peaks[i] - lagtime_peak_samples,
                           peaks < peaks[i]))[0])
        # If there are no peaks lagtime_peak seconds before current peak
        # --> ONSET DETECTED
        if arethey_peaks_before == 0:
            onset_frames = np.append(onset_frames, peaks[i])
    return onset_frames/sr

file_onsets = detect_onsets_from_peaks(abs_sig_diff, peaks1, fe,
                                       lagtime_peak=0.1)
file_onsets2 = detect_onsets_from_peaks(abs_sig_diff, peaks2, fe,
                                       lagtime_peak=0.1)

spf_array1 = np.zeros_like(xprime)

for i in range(len(file_onsets)):
    if i+1 == len(file_onsets):
        next_onset = len(abs_sig_diff)
    else:
        next_onset = file_onsets[i+1]
    
    transient_end = -1
    nb_peaks_cut = nb_peaks2[int(file_onsets[i]*fe):int(next_onset*fe)]
    
    for j in range(len(nb_peaks_cut)-win_len):
        first_sample = int(file_onsets[i]*fe)+j
        last_sample = int(file_onsets[i]*fe)+j+win_len
        # Mean number of peaks in the current windowed nb_peaks function
        mean_peaks = np.mean(nb_peaks2[first_sample:last_sample])
        # Slipping Phase (pseudo)Frequency (spf) in Hz
        spf = fe*mean_peaks/win_len
        # Std
        std_peaks1 = np.std(nb_peaks2[first_sample:last_sample])
        # store both spf and std in respective arrays
        spf_array1[first_sample] = spf
        std_peaks_array[first_sample] = std_peaks1
        
        spf_th_low = 220/2**(1/spf_th_wide)
        spf_th_up = 220*2**(1/spf_th_wide)
        if spf_th_low < spf < spf_th_up:
            if std_peaks1 < std_th:
                transient_end = int(first_sample)  # in samples
                break
    if transient_end == -1:
        continue



# plt.figure()
# plt.plot(np.arange(len(spf_array1))/fe,spf_array1,label = "signal piezzo")


spf_array2 = np.zeros_like(xprime)


for i in range(len(file_onsets2)):
    if i+1 == len(file_onsets2):
        next_onset = len(abs_sig_diff)
    else:
        next_onset = file_onsets2[i+1]
    
    transient_end = -1
    nb_peaks_cut = nb_peaks[int(file_onsets2[i]*fe):int(next_onset*fe)]
    
    for j in range(len(nb_peaks_cut)-win_len):
        first_sample = int(file_onsets2[i]*fe)+j
        last_sample = int(file_onsets2[i]*fe)+j+win_len
        # Mean number of peaks in the current windowed nb_peaks function
        mean_peaks = np.mean(nb_peaks[first_sample:last_sample])
        # Slipping Phase (pseudo)Frequency (spf) in Hz
        spf = fe*mean_peaks/win_len
        # Std
        std_peaks = np.std(nb_peaks[first_sample:last_sample])
        # store both spf and std in respective arrays
        spf_array2[first_sample] = spf
        std_peaks_array[first_sample] = std_peaks
        
        spf_th_low = 220/2**(1/spf_th_wide)
        spf_th_up = 220*2**(1/spf_th_wide)
        if spf_th_low < spf < spf_th_up:
            if std_peaks < std_th:
                transient_end = int(first_sample)  # in samples
                break
    if transient_end == -1:
        continue
    
    


# plt.plot(np.arange(len(spf_array))/fe,spf_array,label = "signal pickup")
# plt.xlabel("temps (s)")
# plt.ylabel("SPF(hz)")
# plt.legend()
# plt.grid()
# plt.show()    

fig,ax = plt.subplots(nrows = 4,sharex = True)

ax[0].plot(t,sig2,label = "signal piezzo")
ax[0].plot(t,sig,label = "signal pickup")

ax[1].plot(t[:511999],xprime,label = "signal piezzo")
ax[1].plot(t[peaks1],xprime[peaks1],'x',label = "signal piezzo")

ax[1].plot(t[:511999],xpick_diff,label = "signal pickup")
ax[1].plot(t[peaks2],xpick_diff[peaks2],'x',label = "signal pickup")

ax[2].plot(np.arange(len(nb_peaks))/fe,nb_peaks,label = "signal pickup")
ax[2].plot(np.arange(len(nb_peaks2))/fe,nb_peaks2,label = "signal piezzo")

ax[3].plot(np.arange(len(spf_array1))/fe,spf_array1,label = "signal piezzo")
ax[3].plot(np.arange(len(spf_array2))/fe,spf_array2,label = "signal pickup")

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()

ax[3].set_xlabel("temps(s)")

ax[0].set_ylabel("x")

ax[1].set_ylabel("|x'|")

ax[2].set_ylabel("y[slips]")

ax[3].set_ylabel("SPF(Hz)")

################################################################################
#transformée de fourier du signal dérivé
#le but est d'identifier la période des pics du signal 
#et donc de vérifier si les deux enveloppes sont superposables
#Le résultat attendu est 3 pics de fréquences principales pour le signal pickup f1,f2,f3
#Et un pic principal pour le signal piezzo à la fréquence f1 où f1 ~ 220 Hz

freqs = np.fft.rfftfreq(N,1/fe)
fft_piezzo = np.abs(np.fft.rfft(xprime))
fft_pickup = np.abs(np.fft.rfft(xpick_diff))

freqs = freqs[:256000]

fig,ax = plt.subplots(nrows=2,sharex=True)

ax[0].plot(freqs,fft_piezzo,label = "fft du piezzo")
ax[1].plot(freqs,fft_pickup,label = "fft du pickup")

ax[0].legend()
ax[1].legend()
ax[0].set_ylabel("amplitude")

ax[1].set_xlabel("fréquence (Hz)")

##############################################################################
#la fréquence du signal x' est bien 220 Hz et on observe bien plusieurs fréquence
#qui restent mystérieuse (pas f2 et f3)
#on cherche donc à déterminer cette fois-ci le décalage temporel entre piezzo et pickup sur le première
#partie du signal
#faudra vérifier si ce décalage est le même sur toutes les périodes et sur tous les fichiers
#imaginer un code qui détermine le décalage temporel entre les deux signaux ?

t1 = 3.2262
t2 = 3.23214
erreur = t2 - t1

#l'erreur est petite mais on remarque que si on décale de 2 pics on pourrait réduire l'erreur.
#comment décaler de 2 pics ?
#décalage de pic ou décalage temporel ? c'est pareil ?
# 2 pics c'est 2 périodes donc 2 x 1/220 secondes donc 200 éléments environ
#on test avec un décalage de 1 période en enlevant les deux premiers pics du signal pickup

#problème, il faut enlever les 2 premiers pics sur CHAQUE périodes transitoires 
#idée: on regarde où sont les DERNIERS pics des transitoires, qui sont ceux avant lesquels y'a rien
# et on gros on met une condition: "si y'a rien avant 2 ou 3 trois x 1/220 s alors les prochains pics
#sont les nouveaux premiers pics du NOUVEAU transitoire.

#idée 2 (plus simple):
#si peaks[i]<peaks[i+1] alors on est dans la première partie de la courbe.
#Il suffit donc d'enlever le 2eme et le 3eme (le premier appartient à la courbe du transitoire précédent)

supp = []
for i in range(len(peaks2)-1):
    if xpick_diff[peaks2[i]]<xpick_diff[peaks2[i+1]]:
        supp.append(i+1)
        
# print(supp)


# filtered_supp = [supp[0]]  # On garde le premier élément de supp

# for i in range(1, len(supp)):  # On commence à partir du deuxième élément
#     if supp[i] >= filtered_supp[-1] + 2:  # Vérifie si la valeur n'est pas consécutive
#         filtered_supp.append(supp[i])

# supp = filtered_supp  # Met à jour supp


 

# print(supp)

tpick = t[peaks2[supp]]
xpick = xpick_diff[peaks2[supp]]





abs_sig_diff3 = np.abs(np.diff(sig))
std_peaks_array3 = np.zeros_like(abs_sig_diff3)

nb_peaks3 = np.zeros_like(abs_sig_diff3)
for i in range(len(abs_sig_diff3) - win_len):
    nb_peaks3[int(win_len/2)+i] = len(np.where(np.logical_and(peaks1>=i,peaks1<=(win_len+i)))[0])

file_onsets3 = detect_onsets_from_peaks(abs_sig_diff, peaks2[supp], fe,
                                       lagtime_peak=0.1)

spf_array3 = np.zeros_like(xprime)

for i in range(len(file_onsets3)):
    if i+1 == len(file_onsets3):
        next_onset = len(abs_sig_diff)
    else:
        next_onset = file_onsets3[i+1]
    
    transient_end = -1
    nb_peaks_cut = nb_peaks3[int(file_onsets3[i]*fe):int(next_onset*fe)]
    
    for j in range(len(nb_peaks_cut)-win_len):
        first_sample = int(file_onsets3[i]*fe)+j
        last_sample = int(file_onsets3[i]*fe)+j+win_len
        # Mean number of peaks in the current windowed nb_peaks function
        mean_peaks3 = np.mean(nb_peaks[first_sample:last_sample])
        # Slipping Phase (pseudo)Frequency (spf) in Hz
        spf3 = fe*mean_peaks3/win_len
        # Std
        std_peaks3 = np.std(nb_peaks3[first_sample:last_sample])
        # store both spf and std in respective arrays
        spf_array3[first_sample] = spf
        std_peaks_array3[first_sample] = std_peaks3
        
        spf_th_low = 220/2**(1/spf_th_wide)
        spf_th_up = 220*2**(1/spf_th_wide)
        if spf_th_low < spf < spf_th_up:
            if std_peaks < std_th:
                transient_end = int(first_sample)  # in samples
                break
    if transient_end == -1:
        continue
        

# fig, ax = plt.subplots(nrows = 4, sharex = True)

# ax[0].plot(t,sig2,label = "signal piezzo")
# ax[0].plot(t,sig,label = "signal pickup")

# ax[1].plot(t[:511999],xprime,label = "signal piezzo")
# ax[1].plot(t[peaks1],xprime[peaks1],'x',label = "signal piezzo")

# ax[1].plot(t[:511999],xpick_diff,label = "signal pickup")
# ax[1].plot(tpick,xpick,'x',label = "signal pickup")

# ax[2].plot(np.arange(len(nb_peaks3))/fe,nb_peaks3,lw= 5,label = "signal pickup")
# ax[2].plot(np.arange(len(nb_peaks2))/fe,nb_peaks2,label= "signal piezzo")

# ax[3].plot(np.arange(len(spf_array3))/fe,spf_array3,label = "signal pickup")
# ax[3].plot(np.arange(len(spf_array1))/fe,spf_array1,label = "signal piezzo")

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()

# ax[3].plot(np.arange(len(spf_array1))/fe,spf_array1,label = "signal piezzo")
# ax[3].plot(np.arange(len(spf_array2))/fe,spf_array2,label = "signal pickup")

# ax[2].plot(np.arange(len(nb_peaks))/fe,nb_peaks,label = "signal pickup")
# ax[2].plot(np.arange(len(nb_peaks2))/fe,nb_peaks2,label = "signal piezzo")



################################################################################"
#on décale l'original de 2 périodes ~ 200 samples




t_delayed = t[peaks2] + 2*1/220


# fig,ax = plt.subplots(nrows = 4,sharex = True)
# ax[0].plot(t,sig2,label = "signal piezzo")
# ax[0].plot(t_delayed,sig,label = "signal pickup")

# ax[1].plot(t[:511999],xprime,label = "signal piezzo")
# ax[1].plot(t[peaks1],xprime[peaks1],'x',label = "signal piezzo")

# ax[1].plot(t_delayed[:511999],xpick_diff,label = "signal pickup delayed")
# ax[1].plot(t_delayed[peaks2],xpick_diff[peaks2],'x',label = "signal pickup delayed")

# ax[2].plot(np.arange(len(nb_peaks))/fe,nb_peaks,label = "signal pickup")
# ax[2].plot(np.arange(len(nb_peaks2))/fe,nb_peaks2,label = "signal piezzo")

# ax[3].plot(np.arange(len(spf_array1))/fe,spf_array1,label = "signal piezzo")
# ax[3].plot(np.arange(len(spf_array2))/fe,spf_array2,label = "signal pickup")

