# How buried are the signals in noise?

Target SNR = -10 dB on every signal — noise power = 10x signal power by design. Demo signals use the global seed=42 realisation; the 250-track music dataset uses per-track sha256(file_id) seeds (re-derived in this script so the noise matches what build_music_dataset.py produced).

**Reading the columns**

- *SNR [dB]* — clean signal power vs total noise power, post ADC clip. The clip removes only the largest noise excursions so this is within 0.01 dB of the spec target.
- *RMS noise / RMS clean* — linear amplitude ratio. At -10 dB SNR this is sqrt(10) ~= 3.16, i.e. the noise is on average 3x louder than the signal in amplitude terms.
- *noise power / signal power* — same idea on a power scale; 10x at -10 dB.
- *peak noisy / peak clean* — capped at 3x by the ADC clip; values at 3 indicate the clip is engaged. The dataset average is below 3 because not every track trips the clip.
- *% time |noise| > |signal|* — instantaneous burial: fraction of samples where the noise amplitude exceeds the clean signal amplitude. This is the visceral 'how often is the signal hidden' number.
- *% time |noise| > 3x |signal|* — deep burial: noise more than triple the local signal.

| signal | fs [Hz] | tracks | SNR [dB] | RMS noise / RMS clean | noise power / signal power | peak noisy / peak clean | % time |noise| > |signal| | % time |noise| > 3x |signal| | clipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG (MIT-BIH r100) | 360 | 1 | -9.89 | 3.12x | 9.7x | 3.00x | 77.4% | 39.9% | yes |
| Vibration (CWRU 97) | 12000 | 1 | -9.99 | 3.16x | 10.0x | 3.00x | 80.5% | 52.3% | yes |
| Music (metal_01, seed 42) | 22050 | 1 | -10.00 | 3.16x | 10.0x | 3.00x | 82.7% | 54.4% | yes |
| Music dataset (250 tracks, all) | 22050 | 250 | -10.00 | 3.16x | 10.0x | 2.85x | 84.2% | 58.3% | 72% of tracks |
|   - jazz (84 tracks) | 22050 | 84 | -10.00 | 3.16x | 10.0x | 2.65x | 83.8% | 58.6% | 50% of tracks |
|   - metal (83 tracks) | 22050 | 83 | -10.00 | 3.16x | 10.0x | 2.91x | 83.8% | 56.6% | 81% of tracks |
|   - pop (83 tracks) | 22050 | 83 | -10.00 | 3.16x | 10.0x | 2.94x | 85.2% | 59.7% | 84% of tracks |

## Key takeaways

- The design point is identical across every signal: -10 dB SNR, noise carrying 10x the signal power.
- The signal is *louder* than the noise on only 15-20 % of samples. The remaining 80 %+ are noise-dominated, and roughly half of all samples sit under noise more than triple the local signal amplitude.
- The ADC clip engages on every demo signal and on 72 % of the 250-track music dataset. Clipping does NOT meaningfully reduce SNR (the post-clip column is essentially unchanged from the pre-clip target) but it does cap peak excursions at 3x clean peak — which is why the classifier story treats clipping as an impulsive-noise event, not a level change.
- This buriedness is the entire reason the rest of the talk exists: a clean-trained classifier sees +99 % accuracy on clean audio and collapses to 34 % on this noise (cross-condition C). Filtering and feature-design claw it back to 80 %.
