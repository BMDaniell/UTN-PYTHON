# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 07:29:24 2020

@author: DANIEL
"""

import testset as sig
import numpy as np
import pylab as plt
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import style
from scipy.ndimage import gaussian_filter
from numpy import arange
import math
from scipy.fftpack  import fft, fftshift
#############  prueba_Dataset  #######################
###################### Señal continua ################
plt.plot(sig.prueba_Dataset,label='Gráfica señal',color='blue')
plt.title('Señal prueba_Dataset')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
#plt.xlim(0,550)
plt.show()
###################### Señal discreta ################
plt.stem(sig.prueba_Dataset,label='Gráfica señal')
plt.title('Señal discreta prueba_Dataset ')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
#plt.xlim(0,200)
plt.show()
################# 
def calc_dft(sig_src_arr):
    sig_dest_imx_arr = [None]*int((len(sig_src_arr)/2))
    sig_dest_rex_arr = [None]*int((len(sig_src_arr)/2))
    sig_dest_mag_arr = [None]*int((len(sig_src_arr)/2))
    
    for j in range(int(len(sig_src_arr)/2)):
        sig_dest_rex_arr[j] =0
        sig_dest_imx_arr[j] =0

    for k in range(int(len(sig_src_arr)/2)):
        for i in range(len(sig_src_arr)):
            sig_dest_rex_arr[k] = sig_dest_rex_arr[k] + sig_src_arr[i]*math.cos(2*math.pi*k*i/len(sig_src_arr))
            sig_dest_imx_arr[k] = sig_dest_imx_arr[k] - sig_src_arr[i]*math.sin(2*math.pi*k*i/len(sig_src_arr))

    for x in range(int(len(sig_src_arr)/2)):
        sig_dest_mag_arr[x] = math.sqrt(math.pow(sig_dest_rex_arr[x],2)+math.pow(sig_dest_imx_arr[x],2))
        
    style.use('ggplot')
    f,plt_arr = plt.subplots(1, sharex=True)
    f.suptitle("Discrete Fourier Transform (DFT)")
    #plt_arr[0].plot(sig_src_arr, color='red')
    #plt_arr[0].set_title("Input Signal",color='red')
    
    #plt_arr[1].plot(sig_dest_rex_arr, color='green')
    #plt_arr[1].set_title("Frequency Domain(Real part)",color='green')

    #plt_arr[2].plot(sig_dest_imx_arr, color='green')
    #plt_arr[2].set_title("Frequency Domain(Imaginary part)",color='green')

    #plt_arr[3].plot(sig_dest_mag_arr, color='magenta')
    #plt_arr[3].set_title("Frequency Domain (Magnitude))",color='magenta')

    plt.xlim(0,200)
    #plt.ylim(0,1000)
    plt.stem(sig_dest_mag_arr)
    
calc_dft(sig.prueba_Dataset)
####################################################
sig_input_prueba_Dataset =sig.prueba_Dataset
freq_domain_signal= fft(sig.prueba_Dataset)
plt.plot(freq_domain_signal,label='fft  prueba_Dataset',color='red')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.xlim(0,1600)
plt.show()
#valor max de x =800
#grafica esta dividida en 8 segmentos
#frecuencia por muestreo es de 100Hz
#200-> 25 Hz
#100-> 12,5 Hz
signal_prueba_Dataset=np.zeros(len(sig.prueba_Dataset))
for i, num in enumerate (sig.prueba_Dataset):
    signal_prueba_Dataset[i]=float(sig.prueba_Dataset[i]/max(sig.prueba_Dataset))
    
plt.plot(signal_prueba_Dataset)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[,181],nyq=1000, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.prueba_Dataset,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos prueba_Dataset ',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[90,181],nyq=1000, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.prueba_Dataset,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 90, 181
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=1000,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.prueba_Dataset,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 90, 181
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=1000,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.prueba_Dataset,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.prueba_Dataset,color='blue')
plt_arr[0].set_title('señal de entrada',color='blue')
plt_arr[1].plot(output,color='red')
plt_arr[1].set_title('señal filtrada pasa bajos',color='red')
plt_arr[2].plot(output_pa,color='green')
plt_arr[2].set_title('señal filtrada pasa altos',color='green')
plt_arr[3].plot(output_pb,color='black')
plt_arr[3].set_title('Señal filtrada pasa banda',color='black')
plt_arr[4].plot(output_rb,color='navy')
plt_arr[4].set_title('Señal filtrada rechaza banda',color='navy')
plt.show()
