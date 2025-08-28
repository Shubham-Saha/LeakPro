import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import ctypes, os
from numpy.ctypeslib import ndpointer
from leakpro.utils.logger import logger

class QGTDecoder:
    """Quantum Group Testing Decoder for full sample matrix"""
    
    def __init__(self):
        """Initialize QGT decoder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'decoder/FedQGT_decoder.so')
        
        # Verify library exists
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Library not found at: {lib_path}\n"
                "Compile with: gcc -shared -o FedQGT_decoder.so -fPIC FedQGT_decoder.c"
            )
        if not os.access(lib_path, os.R_OK):
            raise PermissionError(f"Missing read permissions for: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self.configure_types()

    def configure_types(self):
        """Configure C function signatures"""
        self.lib.BP_decoder.argtypes = [
            ctypes.c_uint16,  # n_input
            ctypes.c_uint16,  # t_input
            ctypes.c_uint8,   # dvmax
            ctypes.c_uint8,   # dcmax
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # vn_input
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # cn_input
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),   # test_outcome_input
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # prevalence_input
            ctypes.c_uint16,  # max_Iter_input
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),   # cn_deg_input
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),   # vn_deg_input
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),   # DEC (output)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # soft_scores (output)
        ]
        self.lib.BP_decoder.restype = None


    # def decode(self, H, test_scores, prevalence=0.1, max_iter=100, threshold=None):
    def qgt_decode(self, H, binary_outcome, prevalence=0.1, max_iter=100):
        # Convert scores to test outcomes
        # test_outcomes = self.convert_test_scores(test_scores, threshold)
        # test_outcomes = self.convert_test_scores(test_scores)
        
        # Run the decoder
        decisions, soft_scores = self.run_decoder(H, binary_outcome, prevalence, max_iter)

        return np.mean(soft_scores)

    def convert_test_scores(self, scores):
        """Convert membership scores to test outcomes for each group"""        
        # Apply threshold to get binary values
        threshold = np.median(scores)
        threshold = round(threshold, 1)
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Test scores: {scores}")
        binary = (scores >= threshold).astype(np.uint8)

        logger.info(f"Binary test outcomes: {binary}")
        
        return binary

    def run_decoder(self, H, test_outcomes, prevalence, max_iter):
        """Run C decoder"""
        # Get matrix dimensions
        groups, samples = H.shape
        
        # Create VN and CN representations
        vn, vn_deg = self.get_vn_representation(H)
        cn, cn_deg = self.get_cn_representation(H)
        
        # Create output buffer
        dec = np.zeros(samples, dtype=np.uint8)
        soft_scores = np.zeros(samples, dtype=np.float64)
        prevalence_vec = np.full(samples, prevalence, dtype=np.float64)
        
        # Run C decoder
        self.lib.BP_decoder(
            ctypes.c_uint16(samples),
            ctypes.c_uint16(groups),
            ctypes.c_uint8(vn.shape[1]),
            ctypes.c_uint8(cn.shape[1]),
            vn.astype(np.int16),
            cn.astype(np.int16),
            test_outcomes.astype(np.uint8),
            prevalence_vec,
            ctypes.c_uint16(max_iter),
            cn_deg.astype(np.uint8),
            vn_deg.astype(np.uint8),
            dec,
            soft_scores
        )

        logger.info(f"Decoder output: {dec}")
        logger.info(f"Soft scores: {soft_scores}")

        return dec, soft_scores

    def get_vn_representation(self, H):
        max_dv = int(np.max(np.sum(H, axis=0)))
        n = H.shape[1]
        vn = -np.ones((n, max_dv), dtype=np.int16)
        vn_deg = np.zeros(n, dtype=np.uint8)
        for j in range(n):
            conn = np.where(H[:, j])[0]
            vn_deg[j] = len(conn)
            vn[j, :len(conn)] = conn
        return vn.astype(np.int16), vn_deg

    def get_cn_representation(self, H):
        max_dc = int(np.max(np.sum(H, axis=1)))
        t = H.shape[0]
        cn = -np.ones((t, max_dc), dtype=np.int16)
        cn_deg = np.zeros(t, dtype=np.uint8)
        for i in range(t):
            conn = np.where(H[i])[0]
            cn_deg[i] = len(conn)
            cn[i, :len(conn)] = conn
        return cn.astype(np.int16), cn_deg


    def process_llr_values(self, llr_values):
        """Process LLR values to produce a final score between 0 and 1 using min-max normalization"""
        
        # Apply min-max normalization to scale values to [0, 1]
        min_llr = np.min(llr_values)
        max_llr = np.max(llr_values)
        
        # Handle the case where all values are the same (min == max)
        if max_llr == min_llr:
            normalized_score = 0.5  # Default to middle value
        else:
            normalized_score = (np.mean(llr_values) - min_llr) / (max_llr - min_llr)
        
        logger.info(f"Min LLR: {min_llr}, Max LLR: {max_llr}")
        logger.info(f"Normalized LLR score: {normalized_score}")
        return normalized_score
    




