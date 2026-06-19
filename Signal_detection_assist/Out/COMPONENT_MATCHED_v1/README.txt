COMPONENT-MATCHED FILTER v1 (stage 7)
========================================================================

Pipeline (all non-oracle — uses only noisy-side information):
  1. Butterworth HP, cutoff=25.0 Hz, order=2, zero-phase
  2. Mains comb — IIR notches at detected frequencies ['51.1', '99.6', '150.7', '199.2'] Hz, Q=35.0
  3. Switching notch — centred at 4403.5 Hz (mean of sidebands ['4403.5']), adaptive Q so width = max(29.4 Hz, 20.0 Hz)
  4. Adaptive Wiener (MS + DD), alpha_p=0.85, alpha_dd=0.98, D=96 frames, B_min=1.3, xi_min=0.01

All notch frequencies are re-discovered from the noisy audio at the
start of each run (low-energy-frame PSD + find_peaks). The spec
(noise.md) is NOT consulted for any numeric coefficient.

Output WAVs go to
  C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered ComponentMatched 7th stage
so that the Training_model cross-condition experiments can use
the stage-7 file set the same way they use stage-6.

Starter-set summary (9 files = 3 x jazz/metal/pop):
  mean noisy  SNR =  -6.25 dB
  mean stage7 SNR =  +2.24 dB (delta +8.50 dB)
  mean noisy  feat-distance = 13.83
  mean stage7 feat-distance = 8.71 (delta -5.13)