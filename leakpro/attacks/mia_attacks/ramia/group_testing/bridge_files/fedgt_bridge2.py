import ctypes
import numpy as np
import os
from numpy.ctypeslib import ndpointer
from leakpro.utils.logger import logger

class GroupTestDecoder:
    """Handles per-sample BCJR decoding with dynamic parity matrices."""
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'decoder/BCJR_4_python.so')
            
        # Critical check: Verify file exists
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at: {lib_path}\n")
        # gcc -shared -o BCJR_4_python.so -fPIC BCJR_4_python.c
            
        # Check permissions (at least read access)
        if not os.access(lib_path, os.R_OK):
            raise PermissionError(
                    f"Missing read permissions for: {lib_path}\n")

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.configure_decoder()

    def configure_decoder(self):
        # Configure C function signature
        p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.lib.BCJR.argtypes = [
            p_ui8_c,  # H
            p_d_c,  # LLRinput
            p_ui8_c,   # test_values
            p_d_c,  # ChannelMatrix
            ctypes.c_double,             # threshold_dec
            ctypes.c_int,                # n_samples # n_clients
            ctypes.c_int,                # n_groups # n_tests
            p_d_c,  # LLRO
            p_ui8_c    # DEC
        ]

    def gt_decode(self, H, binary_outcome, P_MD=0.05, P_FA=0.05):
        """Decode one original sample's representatives."""
        # # Convert scores to binary tests
        # threshold = np.median(test_scores)
        # # Round threshold to 1 decimal place for cleaner comparison
        # threshold = round(threshold, 1)
        # logger.info(f"threshold: {threshold}")
        # test_values = (test_scores >= threshold).astype(np.uint8)

        # logger.info(f"test_scores: {test_scores}")
        # logger.info(f"test_values: {test_values}")

        # Get matrix dimensions
        n_groups, n_samples = H.shape

        # Initialize C buffers
        llr_out = np.zeros((1, n_samples), dtype=np.double)
        dec_out = np.zeros((1, n_samples), dtype=np.uint8)
        channel_matrix = np.array([[1-P_FA, P_FA], [P_MD, 1-P_MD]], dtype=np.double)

        # Validate test_values are group-level (n_groups,) not sample-level
        assert binary_outcome.shape == (n_groups,), \
            f"test_values must be size {n_groups}, got {binary_outcome.shape}"

        # Execute BCJR decoding
        self.lib.BCJR(
            H.astype(np.uint8, order='C'),
            #np.zeros((1, n_samples), dtype=np.double),  # Dummy LLR input
            np.log((1 - 0.1) / 0.1) * np.ones((1, n_samples), dtype=np.double),
            binary_outcome.astype(np.uint8, order='C'),
            channel_matrix,
            0.0,  # threshold_dec
            n_samples, # n_clients,
            n_groups, # n_tests,
            llr_out,
            dec_out
        )

        # flat_llr = llr_out.flatten()
        # Trim, for example, the lowest 20% and highest 20% of LLRs
        # trimmed_scores = np.sort(flat_llr)[int(0.2*len(flat_llr)):int(0.8*len(flat_llr))]
        # final_score = np.mean(trimmed_scores)

        # # final_score = np.mean(llr_out)

        # logger.info(f"Raw LLRs: {llr_out.flatten()}")
        # logger.info(f"Final Score (Mean LLR): {final_score}")

        # return final_score

        # First flatten the array to make it 1D
        flat_llr = llr_out.flatten()
        logger.info(f"Raw llr_out: {llr_out}")
        
        # Define trim percentiles
        trim_lower_percentile = 40 #previous 30
        # trim_upper_percentile = 100
        
        # Sort the flattened array
        sorted_scores = np.sort(flat_llr)
        
        # Calculate quantile indices
        # lower_idx = int(len(flat_llr) * trim_lower_percentile // 100)
        lower_idx = int(len(flat_llr) * trim_lower_percentile // 100)
        # Ensure we always keep at least one to analyze, even for small sample sizes.
        # lower_idx = max(1, int(len(sorted_scores) * trim_lower_percentile / 100))
        
        # upper_idx = int(len(sorted_scores) * trim_upper_percentile // 100)
        
        # Trim bottom and top quantiles using the sorted array
        # trimmed_scores = sorted_scores[lower_idx:upper_idx]
        # trimmed_scores = flat_llr[:lower_idx]
        trimmed_scores = sorted_scores[:lower_idx]
        # logger.info(f"Analyzing the lowest {len(trimmed_scores)} LLRs: {trimmed_scores}")

        # final_score = self.process_llr_values(trimmed_scores)
        # final_score = self.process_llr_values(trimmed_scores)
        final_score = self.process_and_scale_llrs(trimmed_scores)

        return final_score
        # return flat_llr
    
        
    def process_llr_values(self, llr_values):
        # Apply min-max normalization to scale values to [0, 1]
        min_llr = np.min(llr_values)
        max_llr = np.max(llr_values)
        
        # Handle the case where all values are the same (min == max)
        if max_llr == min_llr:
            normalized_score = 0.5  # Default to middle value
            logger.info(f"Normalized LLRs (default 0.5): {normalized_score}")
        else:
            normalized_score = (np.mean(llr_values) - min_llr) / (max_llr - min_llr)
        
        logger.info(f"Min LLR: {min_llr}, Max LLR: {max_llr}")
        logger.info(f"Normalized LLR score: {normalized_score}")
        return normalized_score
    
    def process_and_scale_llrs(self, llr_values):
        """
        Processes a set of LLRs into a single score scaled to [0, 1].
        1. Takes the mean of the provided (already trimmed) LLRs.
        2. Scales this mean value to a [0, 1] range using a fixed scale.
        """
        if llr_values.size == 0:
            return 0.5 # Default score for no evidence

        # First, find the average of the most likely member signals
        mean_llr = np.mean(llr_values)
        
        # Define a fixed, plausible range for LLRs.
        # Any LLR <= LLR_MIN is considered a perfect member (score 1.0)
        # Any LLR >= LLR_MAX is considered a perfect non-member (score 0.0)
        LLR_MIN = 1
        LLR_MAX = 11.0

        # Clip the mean LLR to be within our defined range
        clipped_llr = np.clip(mean_llr, LLR_MIN, LLR_MAX)

        # Scale the clipped LLR to the [0, 1] range.
        # Note the formula: (MAX - value) / (MAX - MIN). This is because a lower LLR
        # should result in a higher final score (closer to 1.0).
        scaled_score = (LLR_MAX - clipped_llr) / (LLR_MAX - LLR_MIN)
        
        return scaled_score