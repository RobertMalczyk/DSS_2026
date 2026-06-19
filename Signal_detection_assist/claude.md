We are about to filter this signal

C:\\Robak\\DSS2026\\Claude\\Signal\_generation\\out\\Model\\noisy



Result of work is stored here

C:\\Robak\\DSS2026\\Claude\\Signal\_detection\_assist\\Out



Please do not implement filtering mechanism that are not discussed with user



User will indicate the strategy



Never use the clean signal as a reference for any filtering step. Filters must be derived from the noisy input alone (or from noise-only statistics, VAD-based estimates, pretrained models, etc.). The clean signal in C:\\Robak\\DSS2026\\Claude\\Signal\_generation\\out\\Model\\clean may be used only for evaluation (computing SNR, feature distance, etc.), never to construct masks, thresholds, or any filter coefficient.

