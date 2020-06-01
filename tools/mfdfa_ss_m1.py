# MFDFA-Analytics-by-SKDataScience
# multifractal DFA singularity spectra - module 01
# Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m1.py
# This code implements a modification of the first-order unifractal analysis algorithm originally described in [1].
# It covers both the detrended fluctuation analysis (DFA) and the Hurst (a.k.a. R/S) analysis methods. For more details
# on the DFA and Hurst analysis methods, please refer to [2, 3].
#
# At the input, 'dx' is a time series of increments of the physical observable 'x(t)', of the length equal to an
# integer power of two greater than two (i.e. 4, 8, 16, 32, etc.), 'normType_p' is any real greater than or
# equal to one specifying the p-norm, 'isDFA' is a boolean value prescribing to use either the DFA-based algorithm or
# the standard Hurst (a.k.a. R/S) analysis, 'normType_q' is any real greater than or equal to one specifying the q-norm.
#
# At the output, 'timeMeasure' is the time measure of the data's support at different scales, 'meanDataMeasure' is
# the data measure at different scales, while 'scales' is the scales at which the data measure is computed.
#
# The conventional way of using the output values is to plot the data measure vs the scales; the time measure,
# being the inverse quantity to the scales, is computed for an alternative representation and may be ignored.
#
# The requirement to have a power-of-two data length is aimed at avoiding inaccuracies when computing the data measure
# on different time scales.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] J.W. Kantelhardt, Fractal and Multifractal Time Series, available at http://arxiv.org/abs/0804.0747, 2008.
# [3] J. Feder, Fractals, Plenum Press, New York, 1988.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.
# Modified by Rian Koja for usage in this project.

import numpy as np


def get_hurst_by_upscaling(dx, norm_type_p=np.inf, is_dfa=1, norm_type_q=1.0):
    # Some initialiation
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)

    # Number of scales involved. P.ss. We use 'round()' to prevent possible malcomputing of the logarithms
    n_scales = np.int(np.round(np.log2(dx_len)))
    j = 2 ** (np.arange(1, n_scales + 1) - 1) - 1
    
    mean_data_measure = np.zeros(n_scales)
    
    # Computing the data measure
    for ji in range(1, n_scales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_left_shift = np.int(dx_k_len / 2)
        dx_right_shift = np.int(dx_k_len / 2)
        
        for k in range(1, n + 1):
            # Get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_with_shifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_left_shift - 1:
                                  k * dx_k_len + dx_shift + dx_right_shift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_with_shifts, np.ones(dx_right_shift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_right_shift - 1:] - j_dx[1 - 1: -dx_right_shift]) / 2.0
            
            # Finally, we compute the range ...
            if norm_type_p == 0:
                rr = np.max(r[2 - 1:]) - np.min(r[2 - 1:])
            elif np.isinf(norm_type_p):
                rr = np.max(np.abs(r[2 - 1:]))
            else:
                rr = (np.sum(r[2 - 1:] ** norm_type_p) / len(r[2 - 1:])) ** (1.0 / norm_type_p)
            # ... and the normalisation factor ("standard deviation")
            ss = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
            if is_dfa == 1:
                ss = 1.0
            
            mean_data_measure[ji - 1] += (rr / ss) ** norm_type_q
        mean_data_measure[ji - 1] = (mean_data_measure[ji - 1] / n) ** (1.0 / norm_type_q)
    
    # We pass from the scales ('j') to the time measure; the time measure at the scale j(n_scales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(n_scales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    time_measure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    return [time_measure, mean_data_measure, scales]
