# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:43:20 2022

@author: Shaowen Chen
"""

import torch
from torch.fft import fft, ifft
import numpy as np
import math
import matplotlib.pyplot as plt
import time as Time
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def ASST(y, nfft, hop, fs, sigma, Fmax=0):
    
    """
    This code is the pytorch-based implement of Aligning Synchrosqueezing Transform (ASST)
    which is the submethod of Statistic Synchrosqueezing Transform (Stat-SST).
    
    Paramters:
        INPUT
        y:       analyzed signal 
        nfft:    number of FFT
        hop:     time shift
        sigma:   sigma of gaussian window
        Fmax:    max band of the time-frequency representation
        
        OUTPUT
        Tx2:     ASST representation
        omega2:  aligning instantaneous frequency (IF) estimator
        stft:    STFT result
        time:    time axis
        freq:    frequency axis
    """
    
    device = y.device
    if Fmax == 0:
        Fmax = fs/2
    B, N = y.size()
    freq=torch.arange(nfft)*fs/nfft
    freqFmax = (freq - Fmax).abs()
    freq = freq[:torch.argmin(freqFmax)].to(device)
    time = torch.arange(0, N, hop)/fs
    K = 0.005
    K = torch.tensor(K)
    w_half = torch.round(torch.sqrt(-2*torch.log(K))*sigma*fs).long()
    len_win = 2*w_half
    if len_win > nfft:
        len_win = int(nfft)
        w_half = int(nfft/2)
    t_win = torch.linspace(-w_half+0.5,w_half+0.5,2*w_half)/fs
    # t_win = (torch.arange(len_win) + 0.5)/fs
    padding = torch.zeros((B, w_half)).to(device)
    y = torch.hstack((padding,y,padding))
    g = (math.pi*sigma**2)**(-0.25)*torch.exp(-(t_win/sigma)**2/2).to(device)
    gp = g*(-t_win/sigma**2).to(y)
    
    
    n_t = time.shape[0]
    f_t = freq.shape[0] 
    stft_R = torch.zeros((B,f_t,n_t),dtype=torch.float32).to(device)
    dstft_R = torch.zeros((B,f_t,n_t),dtype=torch.float32).to(device)
    stft_I = torch.zeros((B,f_t,n_t),dtype=torch.float32).to(device)
    dstft_I = torch.zeros((B,f_t,n_t),dtype=torch.float32).to(device)
    for tim in range(n_t):
        t_index = torch.arange(tim*hop,tim*hop + len_win,1).long()
        data = y[:,t_index] * g
        ddata = y[:,t_index] * gp
        fft_y = (fft(data,nfft)/nfft*2)[:,:torch.argmin(freqFmax)]
        dfft_y = (fft(ddata,nfft)/nfft*2)[:,:torch.argmin(freqFmax)]
        stft_R[...,tim] = fft_y.real 
        dstft_R[...,tim] = dfft_y.real
        stft_I[...,tim] = fft_y.imag 
        dstft_I[...,tim] = dfft_y.imag
    
    stft = torch.zeros((B,f_t,n_t),dtype=torch.complex32).to(device)
    stft = stft_R + 1j * stft_I
    dstft = torch.zeros((B,f_t,n_t),dtype=torch.complex32).to(device)
    dstft = dstft_R + 1j * dstft_I
    
    df = freq[1]-freq[0]
    # F = torch.tile(freq, (stft.shape[-1], 1))
    F = freq.unsqueeze(0).unsqueeze(-1).repeat(B, 1, stft.shape[-1]).to(device)
    omega = -((dstft/stft)).imag / (2 * math.pi)
    omega = F + omega
    gamma = 10**(-8)

    omega2 = torch.zeros((omega.size())).type_as(omega).to(device)
    nt = torch.arange(stft.shape[-1])
    nf = torch.arange(stft.shape[1])
    nb = torch.arange(stft.shape[0])
# modifying the omega
    omegar = (omega - F).abs()    
    s1 = torch.diff(torch.hstack((omegar[:,0,:].unsqueeze(1),omegar)), dim = 1)
    s2 = torch.diff(torch.hstack((omegar,omegar[:,-1,:].unsqueeze(1))), dim = 1)   
    partdw = torch.diff(torch.hstack((omega[:,0,:].unsqueeze(1),omega)), dim = 1).abs()/df
    for b in nb:      
        for ii in nt:
            omegarp = []
            for jj in nf:        
                judge = s1[b,jj,ii]<0 and s2[b,jj,ii]>0 and omegar[b,jj,ii]<df and partdw[b,jj,ii]<1
                if judge:
                    omegarp.append(jj)
                
            range_max = torch.ones(len(omegarp),dtype=torch.int32) * stft.shape[1]
            range_min = torch.zeros(len(omegarp),dtype=torch.int32)    
            for irow in range(len(omegarp)):
                for icol in range(int(omegarp[irow]),0,-1):
                    if partdw[b,icol,ii]>=1:
                        range_min[irow] = icol
                        break
                    # range_min[irow] = 0
                
                for icol in range(int(omegarp[irow]),stft.shape[1],1):
                      if partdw[b,icol,ii]>=1:
                          range_max[irow] = icol
                          break
                      # range_max[irow] = len(nf)
               
                omega2[b,range_min[irow]:range_max[irow],ii] = freq[omegarp[irow]]
    
#####  
    e_modify = torch.exp(1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    e_modify_inv = torch.exp(-1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    stft = stft * e_modify 
    Tx2_R = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    Tx2_I = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    
    for b in nb:        
        for i in nt:
            for j in nf:
                if (stft[b,j,i]).abs()>gamma:
                    k = torch.round(omega2[b,j,i]/df).long()
                    if k>0 and k<len(freq)-1:
                        # Tx[k,i] = Tx[k,i] + tfr_phase[j,i]
                        # Tx2[k,i] = Tx2[k,i] + tfr[j,i]*torch.exp(1j*torch.pi*freq[j])
                        Tx2_R[b,k,i] += stft[b,j,i].real
                        Tx2_I[b,k,i] += stft[b,j,i].imag
                    
    stft = stft * e_modify_inv
    Tx2 = Tx2_R + 1j * Tx2_I
    # Tx2 = Tx2 * e_modify_inv  
    # Tx2 = parallel_Tx_R + 1j * parallel_Tx_I
    Tx2 = Tx2 * e_modify_inv            
    return Tx2,omega2,stft,time,freq

def SST_parellel(y, nfft, hop, fs, sigma, Fmax=0):
    
    """
    This code is the pytorch-based implement of Synchrosqueezing Transform (SST).
    
    Paramters:
        INPUT
        y:       analyzed signal 
        nfft:    number of FFT
        hop:     time shift
        sigma:   sigma of gaussian window
        Fmax:    max band of the time-frequency representation
        
        OUTPUT
        Tx2:     SST representation
        omega2:  aligning instantaneous frequency (IF) estimator
        stft:    STFT result
        time:    time axis
        freq:    frequency axis
    """
    
    device = y.device
    if Fmax == 0:
        Fmax = fs/2
    B, N = y.size()
    freq=torch.arange(nfft)*fs/nfft
    freqFmax = (freq - Fmax).abs()
    freq = freq[:torch.argmin(freqFmax)].to(device)
    time = torch.arange(0, N, hop)/fs
    K = 0.005
    K = torch.tensor(K)
    w_half = torch.round(torch.sqrt(-2*torch.log(K))*sigma*fs).long()
    len_win = 2*w_half
    if len_win > nfft:
    # if len_win < nfft:
        len_win = int(nfft)
        w_half = int(nfft/2)
    t_win = torch.linspace(-w_half+0.5,w_half+0.5,2*w_half)/fs
    # t_win = (torch.arange(len_win) + 0.5)/fs
    padding = torch.zeros((B, w_half)).to(device)
    y = torch.hstack((padding,y,padding))
    g = (math.pi*sigma**2)**(-0.25)*torch.exp(-(t_win/sigma)**2/2).to(device)
    gp = g*(-t_win/sigma**2).to(y)
    
    
    t0 = torch.arange(torch.arange(0, N, hop)[0]*hop,torch.arange(0, N, hop)[0]*hop + len_win,1).long().to(device) 
    time_idx = t0.unsqueeze(-1).repeat(1,time.shape[0]) + torch.arange(0, N, hop).to(device)
    data = (y[:,time_idx].permute(0,2,1) * g).permute(0,2,1)
    ddata = (y[:,time_idx].permute(0,2,1) * gp).permute(0,2,1)
    stft = (fft(data,nfft,dim=1)/nfft*2)[:,:torch.argmin(freqFmax),:]
    dstft = (fft(ddata,nfft,dim=1)/nfft*2)[:,:torch.argmin(freqFmax),:]
    
    df = freq[1]-freq[0]
    # F = torch.tile(freq, (stft.shape[-1], 1))
    F = freq.unsqueeze(0).unsqueeze(-1).repeat(B, 1, stft.shape[-1]).to(device)
    omega = -((dstft/stft)).imag / (2 * math.pi)
    omega = F + omega
    gamma = 10**(-8)       
      
#####  
    e_modify = torch.exp(1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    e_modify_inv = torch.exp(-1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    stft = stft * e_modify 
    # Tx2_R = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    # Tx2_I = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    
  
    size_f = stft.shape[1]
    parallel_Tx_R = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    parallel_Tx_I = torch.zeros((stft.size()),dtype=torch.float32).to(device)
    k = torch.round(omega/df).long()
    k[~(stft.abs()>gamma) | ~(k > 0) | ~(k < (len(freq)-1))] = size_f
    res = torch.zeros((stft.shape[0], size_f+1, stft.shape[-1])).to(device).scatter_add(dim=1, index=k, src=stft.real)
    parallel_Tx_R += res[:,:-1,:]
    res = torch.zeros((stft.shape[0], size_f+1, stft.shape[-1])).to(device).scatter_add(dim=1, index=k, src=stft.imag)
    parallel_Tx_I += res[:,:-1,:]
    
                    
    stft = stft * e_modify_inv
    # Tx2 = Tx2_R + 1j * Tx2_I
    # Tx2 = Tx2 * e_modify_inv  
    Tx2 = parallel_Tx_R + 1j * parallel_Tx_I
    Tx2 = Tx2 * e_modify_inv            
    return Tx2,omega,stft,time,freq

def ASST_parellel(y, nfft, hop, fs, sigma, Fmax=0, fast_inverse: bool = True):
    
    """
    This code is the pytorch-based implement of Aligning Synchrosqueezing Transform (ASST)
    which is the submethod of Statistic Synchrosqueezing Transform (Stat-SST).
    
    Paramters:
        INPUT
        y:                analyzed signal 
        nfft:             number of FFT
        hop:              time shift
        sigma:            sigma of gaussian window
        Fmax:             max band of the time-frequency representation
        fast_inverse:     choose the reconstruction way, if "True", use function "iASST", 
                          if "False", use function "iASST_2"
        
        OUTPUT
        Tx2:              ASST representation
        omega2:           aligning instantaneous frequency (IF) estimator
        stft:             STFT result
        time:             time axis
        freq:             frequency axis
    """
    
    device = y.device
    if Fmax == 0:
        Fmax = fs/2
    B, N = y.size()
    freq=torch.arange(nfft)*fs/nfft
    freqFmax = (freq - Fmax).abs()
    freq = freq[:torch.argmin(freqFmax)].to(device)
    time = torch.arange(0, N, hop)/fs
    K = 0.005
    K = torch.tensor(K)
    w_half = torch.round(torch.sqrt(-2*torch.log(K))*sigma*fs).long()
    len_win = 2*w_half
    if len_win > nfft:
        len_win = int(nfft)
        w_half = int(nfft/2)
    t_win = torch.linspace(-w_half+0.5,w_half+0.5,2*w_half)/fs
    # t_win = (torch.arange(len_win) + 0.5)/fs
    padding = torch.zeros((B, w_half)).to(device)
    y = torch.hstack((padding,y,padding))
    g = (math.pi*sigma**2)**(-0.25)*torch.exp(-(t_win/sigma)**2/2).to(device)
    gp = g*(-t_win/sigma**2).to(y)

    
    t0 = torch.arange(torch.arange(0, N, hop)[0]*hop,torch.arange(0, N, hop)[0]*hop + len_win,1).long().to(device) 
    time_idx = t0.unsqueeze(-1).repeat(1,time.shape[0]) + torch.arange(0, N, hop).to(device)
    data = (y[:,time_idx].permute(0,2,1) * g).permute(0,2,1)
    ddata = (y[:,time_idx].permute(0,2,1) * gp).permute(0,2,1)
    stft = (fft(data,nfft,dim=1)/nfft*2)[:,:torch.argmin(freqFmax),:]
    dstft = (fft(ddata,nfft,dim=1)/nfft*2)[:,:torch.argmin(freqFmax),:]
    
    df = freq[1]-freq[0]
    # F = torch.tile(freq, (stft.shape[-1], 1))
    F = freq.unsqueeze(0).unsqueeze(-1).repeat(B, 1, stft.shape[-1]).to(device)
    omega = -((dstft/stft)).imag / (2 * math.pi)
    omega = F + omega
    gamma = 10**(-8)

    omega2 = torch.zeros((omega.size())).type_as(omega).to(device)
# modifying the omega
    omegar = (omega - F).abs()    
    s1 = torch.diff(torch.hstack((omegar[:,0,:].unsqueeze(1),omegar)), dim = 1)
    s2 = torch.diff(torch.hstack((omegar,omegar[:,-1,:].unsqueeze(1))), dim = 1)   
    partdw = torch.diff(torch.hstack((omega[:,0,:].unsqueeze(1),omega)), dim = 1).abs()/df
   
    
    # select out the desirable points 
    judge = (s1<0) * (s2>0) * (omegar<df) * (partdw<1)       
    # get their indices   
    indices = torch.nonzero(judge)
    # parallel version  
    bb,ff,tt = indices[:,0],indices[:,1],indices[:,2]
    mask1 = (partdw[bb,:,tt]>1)
    mask1[:,-1] = True
    range_max = torch.argmin(torch.where(mask1 & (freq >= omega[bb,ff,tt,None]),freq,torch.full_like(freq, freq[-1] + df)),dim=1)
    range_min = torch.argmax(torch.where((partdw[bb,:,tt]>1) & (freq <= omega[bb,ff,tt,None]),freq,torch.full_like(freq,   0)),dim=1)
    # fill between WLR(t) and WHR(t) with WR(t) by using cumsum
    
    tmp = torch.zeros_like(omega2)
    tmp[bb,range_min,tt] += omega[bb,ff,tt]
    tmp[bb,range_max,tt] -= omega[bb,ff,tt]
    omega2 = torch.cumsum(tmp,dim=1)          
      
#####  
    e_modify = torch.exp(1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    e_modify_inv = torch.exp(-1j*2*math.pi*w_half/fs*freq).unsqueeze(0).unsqueeze(-1).repeat(stft.shape[0], 1, stft.shape[-1])
    stft = stft * e_modify 

    Tx = torch.zeros((stft.size())).type_as(stft).to(device)
    k = torch.round(omega2/df).long()
    k[~(stft.abs()>gamma) | ~(k > 0) | ~(k < (len(freq)-1))] = stft.shape[1]
    res = torch.zeros((stft.shape[0], stft.shape[1]+1, stft.shape[-1])).type_as(stft).to(device).scatter_add(dim=1, index=k, src=stft)
    Tx += res[:,:-1,:]
                    
    stft = stft * e_modify_inv
    if fast_inverse:
        Tx = Tx * e_modify_inv            
    return Tx,omega2,stft,time,freq



def iASST(Tx,nfft,fs,sigma,hop,N,Fmax=0):
    
    """
    This the fast version for reconstructing the signal from ASST or SST representation.
    If "hop" is bigger than 1, use this code to reconstruct the signal.
    
    Parameters:
        
        INPUT
        Tx:       ASST or SST representation
        nfft:     number of FFT
        fs:       sampling rate
        sigma:    sigma of gaussian window
        hop:      time shift
        N:        length of signal
        Fmax:     max band of the time-frequency representation
        
        OUTPUT
        Re:       reconstructed signal
    """
    
    device = Tx.device
    if Fmax == 0:
        Fmax = fs/2
    b_size, f_size, t_size = Tx.size()    
    freq=torch.arange(nfft)*fs/nfft
    freqFmax = (freq - Fmax).abs()
    freq_interval = freq.shape[0]/2 - torch.argmin(freqFmax) 
        
    K = 0.005
    K = torch.tensor(K)
    w_half = torch.round(torch.sqrt(-2*torch.log(K))*sigma*fs).long()
    len_win = 2*w_half
    if len_win> nfft:
        len_win = nfft
        w_half = int(nfft/2)
    t_win = torch.linspace(-w_half+0.5,w_half+0.5,2*w_half)/fs
    g = (math.pi*sigma**2)**(-0.25)*torch.exp(-(t_win/sigma)**2/2).to(device)
    # c1 = sum((torch.pi*sigma**2)**(-0.5)*torch.exp(-(t_win/sigma)**2/2))/fs
    c = torch.sum((math.pi*sigma**2)**(-0.5)*torch.exp(-(t_win/sigma)**2/2))/fs  
    Tx = torch.cat((Tx,torch.zeros((b_size,int(nfft/2+freq_interval),t_size)).to(device)), dim = 1)
    # Tx_re = torch.zeros(shape=(int(len_win),int(Tx.shape[1])))
    Tx_re = torch.zeros(Tx.size())
    Tx_re = ifft(Tx,nfft,dim=1).real
    y = torch.zeros((b_size,len_win,(t_size-1)*hop+len_win)).type_as(Tx_re)       
    
    idx0 = torch.arange(0,t_size*hop,hop).to(device)
    win_index = (idx0.unsqueeze(-1).repeat(1,len_win) + torch.arange(len_win).to(device)).T.unsqueeze(0).repeat(b_size,1,1)
    res = (g * Tx_re[:,:len_win,:].permute(0,2,1)).permute(0,2,1)
    y.scatter_(2,win_index,res)
    Re = torch.sum(y,dim=1)
    Re = Re[:,int(w_half):int(w_half)+N]*nfft/fs*hop/c    
    return Re

def iASST_2(Tx,sigma):
    
    """
    This the fast version for reconstructing the signal from ASST or SST representation.
    If "hop" is equal to 1, use this code to reconstruct the signal.
    
    Parameters:
        
        INPUT
        Tx:       ASST or SST representation
        sigma:    sigma of gaussian window
        
        OUTPUT
        Re:       reconstructed signal
    """
    
    device = Tx.device
    Re = Tx.real.sum(dim = 1)/((math.pi*sigma**2)**(-0.25)*torch.exp(-(torch.tensor(0)/sigma)**2/2)).to(device)
    return Re


if __name__ == "__main__":
    plt.close("all")
    fs = 1024
    hop = 1
    nfft = fs
    sigma = 0.08
    t = np.arange(0,10*fs)/fs
    fmax = 100
    pi=math.pi
    Sig = np.cos(2*pi*50*t) + np.cos(2*pi*(17*t + 6*np.sin(1.5*t))) + np.cos(2*pi*(40*t + 1*np.sin(1.5*t)))
    Sig = torch.from_numpy(Sig).to("cuda:0").unsqueeze(0)
    Tx2,omega2,stft,time,freq = ASST_parellel(Sig, nfft, hop, fs, sigma, Fmax = fmax,fast_inverse=False)
    # Tx2,omega2,stft,time,freq = ASST_fast(speech, nfft, hop, fs, sigma)

    Re = iASST(Tx2,nfft,fs,sigma,hop,Sig.shape[-1])
    Re = iASST_2(Tx2,sigma)
    sig = Sig.data.cpu().numpy()
    plt.figure()
    plt.plot(sig[0])
    
    plt.figure()
    plt.pcolormesh(time.data.cpu().numpy(),freq.data.cpu().numpy(),(omega2[0,...].data.cpu().numpy()))
    
    plt.figure()
    plt.pcolormesh(time.data.cpu().numpy(),freq.data.cpu().numpy(),abs(Tx2[0,...].data.cpu().numpy()))
    
    plt.figure()
    plt.pcolormesh(time.data.cpu().numpy(),freq.data.cpu().numpy(),abs(stft[0,...].data.cpu().numpy()))
    
    
   
