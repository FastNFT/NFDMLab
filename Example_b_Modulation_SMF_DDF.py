import Examples
import sys
import numpy as np
import scipy.io as scio

Fiber_type = 'DDF'  # 'DDF' or 'SSMF'

if Fiber_type == 'DDF':
    example = Examples.GuiEtAl2018_DDF()   
    example.n_steps_per_span = 500   
    example.path_average = False
    example.beta2 = -25.491e-27
    example.gamma = 1.3e-3
elif Fiber_type == 'SSMF':
    example = Examples.GuiEtAl2018()
    example.n_steps_per_span = 200
    Factor_gamma_eff_80km = 0.2646
    example.beta2 = -25.49e-27*Factor_gamma_eff_80km  #-8.414e-27
    example.gamma = 1.3e-3
    example.path_average = True
else:
    raise Exception('Specify the correct fiber type')
example.n_spans = 1
#example.gamma = 1.3e-3     #2.2e-3
#example.alpha = 0
example.constellation_type = 'PSK'
example.constellation_level = 8
example.n_symbols_per_block = 9
example.fiber_span_length =  80e3
example.noise = True
example.noise_figure = 6
example.tx_bandwidth = 40e9
example.rx_bandwidth = 40e9
example.Ed = 16
example.reconfigure()
runs = 10
tx_data, rx_data = example.run(runs)


#Rotation compensation
###########################################################################################
#Compensate for the rotation mismatch
RotationCompensation = 1
if RotationCompensation == 1:
     #Find the average rotation error
     Rotation_Error = (np.angle(rx_data["symbols"])-np.angle(tx_data["symbols"])-np.pi)%(2*np.pi)+np.pi
     MeanRotation = np.mean(Rotation_Error)
     #Compensate for the average rotation
     rx_data["symbols"] = rx_data["symbols"] * np.exp(-1j*MeanRotation)
###########################################################################################

#Plot the result

example.evaluate_results(tx_data, rx_data)


print('Energy per carrier',example.Ed)
print('Fiber type : ', Fiber_type)
