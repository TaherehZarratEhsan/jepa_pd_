import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.linear_model import LinearRegression


class feature_ext_analysis:
    def __init__(self, config):
        self.config = config

    def _extract_features(self, distances, fps):
         distances_ = np.array(distances)

         if self.config['test_type']=='la':
            distances_ = distances - np.mean(distances)
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  5.0
            normal_cutoff = cutoff_frequencies / nyq  # normalize the cutoff
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=7 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=7, height=-np.mean(distances_), prominence=np.mean(distances_)/2)

         elif self.config['test_type']=='ft':
         
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  9.0
            normal_cutoff = cutoff_frequencies / nyq  # normalize the cutoff
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=5 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=5, height=-np.mean(distances_), prominence=np.mean(distances_)/2)
                
         
            
         #peaks, troughs = self.remove_consecutive_events(distances_, peaks, troughs)
         ############################################################## Compute speed signal
         time_interval = 1 / fps
         speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime

         ###################################################  amp
        
         
         while len(peaks) > 0 and len(troughs) > 0 and troughs[0] > peaks[0]:
             peaks = peaks[1:]
         
         amplitudes = []
         amp_frame_numbers = []

         for i in range(min(len(peaks), len(troughs))):
             peak = peaks[i]
             valid_troughs = troughs[troughs < peak]
             if len(valid_troughs) == 0:
                 continue
             last_trough = valid_troughs[-1]
             amp = abs(distances_[peak] - distances_[last_trough])
             amplitudes.append(amp)
             amp_frame_numbers.append(peak)


         amp_frame_numbers = np.array(amp_frame_numbers).reshape(-1, 1)
         # Compute median and max amplitude
         median_amplitude = np.median(amplitudes)
         max_amplitude = np.max(amplitudes)
         avg_amplitude = np.mean(amplitudes)


         time_points_amp = np.arange(len(amplitudes)).reshape(-1, 1)
         model_amp = LinearRegression()
         model_amp.fit(time_points_amp, amplitudes)
         #model_amp.fit(amp_frame_numbers, amplitudes)

         amp_slope = model_amp.coef_[0]######################################################################################

         ################################################################# Generate per-cycle speed 
         per_cycle_speed_maxima = []
         per_cycle_speed_avg = []
         per_cycle_speed_avg_frame_numbers = []
         for i in range(len(amplitudes) - 1):
             start_idx = peaks[i]  # Start of the window
             end_idx = peaks[i + 1]  # End of the window
             window_speed = speed_signal[start_idx:end_idx]  # Slice the speed signal
             
             if len(window_speed) > 0:  # Ensure the window is not empty
                 per_cycle_speed_maxima.append(np.percentile(np.abs(window_speed), 95))
                 per_cycle_speed_avg.append(np.mean(np.abs(window_speed)))
                 # mid-frame to regress the average values (just for viz/trend)
                 avg_frame = (start_idx + end_idx) // 2
                 per_cycle_speed_avg_frame_numbers.append(avg_frame)
         
            # Compute the median and max of per-cycle speed maxima
         mean_percycle_max_speed = np.mean(per_cycle_speed_maxima)
         mean_percycle_avg_speed = np.mean(per_cycle_speed_avg)
         per_cycle_speed_avg_frame_numbers = np.array(per_cycle_speed_avg_frame_numbers).reshape(-1, 1)
         avg_speed = np.mean(np.abs(speed_signal))
         
         time_points_speed = np.arange(len(per_cycle_speed_avg)).reshape(-1, 1)
         model_speed = LinearRegression()
         model_speed.fit(time_points_speed, np.abs(per_cycle_speed_avg))
         #model_speed.fit(per_cycle_speed_avg_frame_numbers, np.abs(per_cycle_speed_avg))

         speed_slope = model_speed.coef_[0]#####################################################################################
         # Compute tapping intervals (time between consecutive maxima)
         tapping_intervals = np.diff(peaks) / fps

         median_tapping_interval = np.median(tapping_intervals)
         mean_tapping_interval = np.mean(tapping_intervals)

         time_points_ti = np.arange(len(tapping_intervals)).reshape(-1, 1)
         model_ti = LinearRegression()
         model_ti.fit(time_points_ti, tapping_intervals)
         ti_slope = model_ti.coef_[0]


         
         
         ################################## ratio decrement
         mid = len(amplitudes) // 2
         first_half = amplitudes[:mid]
         second_half = amplitudes[mid:]
         mean_first = np.mean(first_half)
         mean_second = np.mean(second_half)
         amp_decrement_ratio = mean_second / mean_first 
         mid = len(np.abs(speed_signal)) // 2
         first_half = np.abs(speed_signal)[:mid]
         second_half = np.abs(speed_signal)[mid:]
         mean_first = np.mean(first_half)
         mean_second = np.mean(second_half)
         speed_decrement_ratio = mean_second / mean_first 
         #############################  new amp ti decrement 
         

         

         
         ####################################################################   hesitation-halts

         std_tapping_intervals = np.std(tapping_intervals)
         cov_tapping_interval = std_tapping_intervals/mean_tapping_interval

         std_amp = np.std(amplitudes)
         cov_amp = std_amp/avg_amplitude

         std_per_cycle_speed_maxima = np.std(per_cycle_speed_maxima)
         cov_per_cycle_speed_maxima = std_amp/mean_percycle_max_speed

         std_per_cycle_speed_avg = np.std(per_cycle_speed_avg)
         cov_per_cycle_speed_avg = std_per_cycle_speed_avg/mean_percycle_avg_speed
         

         std_speed = np.std(speed_signal)
         cov_speed = std_speed/avg_speed



         # Compute total number of interruptions
         threshold = 2 * median_tapping_interval
         num_interruptions1 = sum(interval > threshold for interval in tapping_intervals)
         
         threshold = 1.5 * median_tapping_interval
         num_interruptions2 = sum(interval > threshold for interval in tapping_intervals)
         ################################################################################################

         # FFT-based maximum magnitude feature (dominant frequency component magnitude)
         n = len(distances_)
         fft_spectrum = np.abs(np.fft.rfft(distances_))
         fft_spectrum_no_dc = fft_spectrum[1:]  # Exclude DC component (index 0)
         max_freq_magnitude = np.max(fft_spectrum_no_dc)

         #  Hjorth Parameters
         first_deriv = np.diff(distances_)
         var_zero = np.var(distances_)
         var_d1 = np.var(first_deriv)
         hjorth_mob = np.sqrt(var_d1 / var_zero) #if var_zero > 0 else 0
             
         '''
         #########################################################################################
         # 1. AMPLITUDE DECAY (Exponential instead of linear slope)
        #################################################################
         from scipy.optimize import curve_fit
 
        # Define exponential decay function
         def exp_decay(t, a, λ, c):
            return a * np.exp(-λ * t) + c
         
         time_points_amp = amp_frame_numbers
         time_points_amp = np.arange(len(amplitudes))
         
         a0_amp = max(0.1, amplitudes[0])
         c0_amp = max(0.1, amplitudes[-1])
         if amplitudes[0] > 0 and amplitudes[-1] > 0:
            ratio = amplitudes[-1] / amplitudes[0]
            λ0_amp = -np.log(max(0.01, ratio)) / max(1, len(amplitudes) - 1)
            λ0_amp = min(5, max(0.001, λ0_amp))  # Bound between 0.001 and 5
         else:
            λ0_amp = 0.05
         
        # Fit exponential decay - NO FALLBACK
         popt_amp, _ = curve_fit(exp_decay, time_points_amp, amplitudes, 
                       p0=[a0_amp, λ0_amp, c0_amp], 
                       bounds=([0.001, 0.0001, 0], [100, 5, 100]),
                       maxfev=10000)  # Increased from default
         
         a_fit_amp, λ_fit_amp, c_fit_amp = popt_amp
         amp_decay_rate = λ_fit_amp
         amp_fitted_line = exp_decay(time_points_amp, a_fit_amp, λ_fit_amp, c_fit_amp)
    
   
         # ========== EXPONENTIAL DECAY FOR SPEED ==========
         time_points_speed = per_cycle_speed_avg_frame_numbers.squeeze()
         time_points_speed = np.arange(len(per_cycle_speed_avg))
         speed_values = np.abs(per_cycle_speed_avg)
        
         
         # Safe initialization with simple guesses
         a0_speed = max(0.1, speed_values[0]) if len(speed_values) > 0 else 1.0
         c0_speed = max(0.1, speed_values[-1]) if len(speed_values) > 0 else 0.1
         λ0_speed = 0.05  # Simple guess
                
         # Fit exponential decay - NO FALLBACK
         popt_speed, _ = curve_fit(exp_decay, time_points_speed, speed_values,
                                 p0=[a0_speed, λ0_speed, c0_speed],
                                 bounds=([0.001, 0.0001, 0], [1000, 5, 1000]),
                                 maxfev=10000)
         
         a_fit_speed, λ_fit_speed, c_fit_speed = popt_speed
         speed_decay_rate = λ_fit_speed
         speed_fitted_line = exp_decay(time_points_speed, a_fit_speed, λ_fit_speed, c_fit_speed)
       
 
         
         # ========== EXPONENTIAL GROWTH FOR CYCLE DURATION ==========
         def exp_growth(t, a, λ, c):
            return a * np.exp(λ * t) + c
         
         time_points_tap = np.arange(len(tapping_intervals))
                    
         a0_tap = max(0.1, tapping_intervals[0])
         c0_tap = max(0.1, tapping_intervals[-1])
         λ0_tap = 0.05  # Simple guess for growth rate
        
         # Fit exponential growth with increased maxfev
         popt_tap, _ = curve_fit(exp_growth, time_points_tap, tapping_intervals,
                               p0=[a0_tap, λ0_tap, c0_tap],
                               bounds=([0.001, 0.0001, 0], [10, 5, 10]),
                               maxfev=10000)
         a_fit_tap, λ_fit_tap, c_fit_tap = popt_tap
         duration_growth_rate = λ_fit_tap
         fitted_tapping_line = exp_growth(time_points_tap, a_fit_tap, λ_fit_tap, c_fit_tap)
 
         '''
         features = {'avg_amplitude': avg_amplitude,
                     'mean_percycle_max_speed':mean_percycle_max_speed,
                     'mean_percycle_avg_speed':mean_percycle_avg_speed,
                     'mean_tapping_interval':mean_tapping_interval,
                     'amp_slope':amp_slope, 
                     'ti_slope':ti_slope,
                     'speed_slope':speed_slope,
                     'cov_tapping_interval':cov_tapping_interval, 
                     'cov_amp':cov_amp, 
                     'cov_per_cycle_speed_maxima':cov_per_cycle_speed_maxima,
                     'cov_per_cycle_speed_avg':cov_per_cycle_speed_avg,
                     'num_interruptions2':num_interruptions2,
                     'hjorth_mob':hjorth_mob, 
                     'max_freq_magnitude':max_freq_magnitude, 
                     }

  
         
         return list(features.values()),  list(features.keys())
