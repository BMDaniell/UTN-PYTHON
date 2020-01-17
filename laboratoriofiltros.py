# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:36:00 2020

@author: DANIEL
"""

import Datos_filtrar as sig
import numpy as np
import pylab as plt
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import style
from scipy.ndimage import gaussian_filter
from numpy import arange
import math
from scipy.fftpack  import fft, fftshift
#############  noise_1KHz  #######################
####################################################
sig_input_noise_1KHz =sig.noise_1KHz
freq_domain_signal= fft(sig.noise_1KHz)
plt.plot(freq_domain_signal,label='fft  noise_1KHz',color='red')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.xlim(0,550)
plt.show()
#valor max de x =275
#grafica esta dividida en 5,5 segmentos
#frecuencia por muestreo es de 1Khz-> 1000 Hz
#100-> 181,81 Hz
#50-> 90,91 Hz
signal_noise_1KHz=np.zeros(len(sig.noise_1KHz))
for i, num in enumerate (sig.noise_1KHz):
    signal_noise_1KHz[i]=float(sig.noise_1KHz[i]/max(sig.noise_1KHz))
    
plt.plot(signal_noise_1KHz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[90,181],nyq=1000, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.noise_1KHz,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos noise_1KHz ',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[90,181],nyq=1000, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.noise_1KHz,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 90, 181
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=1000,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.noise_1KHz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 90, 181
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=1000,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.noise_1KHz,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.noise_1KHz,color='blue')
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



#############  noise_100Hz  #######################
####################################################
sig_input_noise_100Hz  =sig.noise_100Hz
freq_domain_signal= fft(sig.noise_100Hz)
plt.plot(freq_domain_signal,label='fft  noise_100Hz',color='blue')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.show()
#valor max de x =175
#grafica esta dividida en 7 segmentos
#frecuencia por muestreo es de 100hz
#50-> 14,5Hz
#25-> 7,25Hz
signal_noise_100Hz=np.zeros(len(sig.noise_100Hz))
for i, num in enumerate (sig.noise_100Hz):
    signal_noise_100Hz[i]=float(sig.noise_100Hz[i]/max(sig.noise_100Hz))
    
plt.plot(signal_noise_100Hz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[7,15],nyq=100, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.noise_100Hz,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos noise_100Hz)',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[7,15],nyq=100, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.noise_100Hz,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 7, 15
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=100,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.noise_100Hz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 7, 15
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=100,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.noise_100Hz,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.noise_100Hz,color='blue')
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

#############  extra_noise_100Hz #######################
####################################################
sig_input_extra_noise_100Hz =sig.extra_noise_100Hz 
freq_domain_signal= fft(sig.extra_noise_100Hz )
plt.plot(freq_domain_signal,label='fft extra_noise_100Hz',color='green')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.show()
#valor max de x =200
#grafica esta dividida en 8 segmentos
#frecuencia por muestreo es de 100Hz
#50-> 12,5Hz
#25-> 6,25Hz
signal_extra_noise_100Hz =np.zeros(len(sig.extra_noise_100Hz ))
for i, num in enumerate (sig.extra_noise_100Hz):
    signal_extra_noise_100Hz [i]=float(sig.extra_noise_100Hz[i]/max(sig.extra_noise_100Hz))
    
plt.plot(signal_extra_noise_100Hz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()
#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[6,13],nyq=100, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.extra_noise_100Hz ,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos extra_noise_100Hz ',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[6,13],nyq=100, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.extra_noise_100Hz ,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 6, 13
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=100,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.extra_noise_100Hz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 6, 13
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=100,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.extra_noise_100Hz ,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.extra_noise_100Hz ,color='blue')
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


#############  ecg_100Hz  #######################
####################################################
sig_input_ecg_100Hz =sig.ecg_100Hz
freq_domain_signal= fft(sig.ecg_100Hz)
plt.plot(freq_domain_signal,label='fft ecg_100Hz  ',color='orange')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.show()
#valor max de x =300
#grafica esta dividida en 6 segmentos
#frecuencia por muestreo es de 100hz
#100-> 16,67Hz
#50-> 8,33Hz
signal_ecg_100Hz =np.zeros(len(sig.ecg_100Hz ))
for i, num in enumerate (sig.ecg_100Hz):
    signal_ecg_100Hz [i]=float(sig.ecg_100Hz[i]/max(sig.ecg_100Hz))
    
plt.plot(signal_ecg_100Hz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[8,17],nyq=100, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.ecg_100Hz,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos Ecg_100Hz',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[8,17],nyq=100, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.ecg_100Hz,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 8, 17
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=100,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.ecg_100Hz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 8, 17
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=100,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.ecg_100Hz,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.ecg_100Hz,color='blue')
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

############# airflow_calibrated_100Hz  #######################
####################################################
sig_input_airflow_calibrated_100Hz =sig.airflow_calibrated_100Hz
freq_domain_signal= fft(sig.airflow_calibrated_100Hz)
plt.plot(freq_domain_signal,label='fft  airflow_calibrated_100Hz',color='magenta')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.xlim(0,910)
plt.show()
#valor max de x =455
#grafica esta dividida en 5 segmentos
#frecuencia por muestreo es de 100hz
#200-> 21,98Hz
#91-> 10Hz
signal_airflow_calibrated_100Hz =np.zeros(len(sig.airflow_calibrated_100Hz))
for i, num in enumerate (sig.airflow_calibrated_100Hz):
    signal_airflow_calibrated_100Hz[i]=float(sig.airflow_calibrated_100Hz[i]/max(sig.airflow_calibrated_100Hz))
    
plt.plot(signal_airflow_calibrated_100Hz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[10,22],nyq=100, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.airflow_calibrated_100Hz,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos airflow_calibrated_100Hz',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[10,22],nyq=100, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.airflow_calibrated_100Hz,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 10, 22
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=100,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.airflow_calibrated_100Hz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 10, 22
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=100,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.airflow_calibrated_100Hz,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.airflow_calibrated_100Hz,color='blue')
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

#############  airflow_100Hz  #######################
####################################################
sig_input_airflow_100Hz =sig.airflow_100Hz
freq_domain_signal= fft(sig.airflow_100Hz)
plt.plot(freq_domain_signal,label='fft  airflow_100Hz ',color='blue')
plt.title('Transformada Rapida de Fourier')
plt.legend(loc='best')
plt.grid(True,color='gray',linestyle='--')
plt.show()
#valor max de x =700
#grafica esta dividida en 7 segmentos
#frecuencia por muestreo es de 100hz
#200-> 14,28Hz
#100-> 7,14Hz
signal_airflow_100Hz =np.zeros(len(sig.airflow_100Hz))
for i, num in enumerate (sig.airflow_calibrated_100Hz):
    signal_airflow_100Hz[i]=float(sig.airflow_100Hz[i]/max(sig.airflow_100Hz))
    
plt.plot(signal_airflow_100Hz)
plt.title('Señal original con reducción de escala')
plt.grid(True,color='gray',linestyle='--')
plt.show()

#___________________ filtros_______________#

##Filtro pasa bajos####

lowpass_coef=signal.firwin(31,[7,15],nyq=100, window='nuttall')
#convolucion entre la señal de entrada y el kernel 
output=signal.convolve(sig.airflow_100Hz,lowpass_coef, mode='same')
plt.plot(output,label='Filtro pasa bajos',color='peru')
plt.title('Filtro Pasa Bajos Ecg_100Hz',color='peru')
plt.legend(loc='center right')
plt.grid(True,color='black',linestyle='--')
plt.show()
####filtro pasa altos####

highpas_coef=signal.firwin(31,[7,15],nyq=100, pass_zero=False,window='blackman')#ganacia
output_pa=signal.convolve(sig.airflow_100Hz,highpas_coef, mode='same')

###Filtro pasa banda###
f1, f2 = 7, 15
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=False ,nyq=100,window='nuttall')#pasa banda
output_pb=signal.convolve(sig.airflow_100Hz,bandpass_coef, mode='same')

###Filtro rechaza banda###
f1, f2 = 7, 15
bandpass_coef=signal.firwin(19, [f1, f2], pass_zero=True ,nyq=100,window='nuttall')#pasa banda
output_rb=signal.convolve(sig.airflow_100Hz,bandpass_coef, mode='same')

##Graficas filtros 
f,plt_arr=plt.subplots(5,sharex=True)
plt_arr[0].plot(sig.airflow_100Hz,color='blue')
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