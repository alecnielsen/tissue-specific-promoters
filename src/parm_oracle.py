"""
PARM (Promoter Activity Regulatory Model) oracle wrapper for HEK293/HEK116 predictions.

PARM provides pretrained CNN models for predicting promoter activity from DNA sequences.
We use the HEK116 model as a proxy for HEK293 (both are human embryonic kidney cell lines).

Reference: https://github.com/vansteensellab/PARM
Paper: https://www.biorxiv.org/content/10.1101/2024.07.09.602649v1
"""

import tempfile
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np


class PARMOracle:
    """
    Wrapper for PARM model to predict HEK293-like promoter activity.

    Uses the pretrained HEK116 model from PARM as a proxy for HEK293.
    Both are human embryonic kidney cell lines with similar characteristics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        use_gpu: bool = True,
    ):
        """
        Initialize PARM oracle.

        Args:
            model_path: Path to PARM model directory (default: auto-detect)
            batch_size: Batch size for predictions
            use_gpu: Whether to use GPU for predictions
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self._check_parm_available()

    def _check_parm_available(self):
        """Check if PARM is installed and available."""
        try:
            result = subprocess.run(
                ["parm", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("PARM command failed")
        except FileNotFoundError:
            raise RuntimeError(
                "PARM not found. Install with: conda install -c bioconda parm"
            )

    def score_sequences(self, sequences: list[str]) -> np.ndarray:
        """
        Score DNA sequences using PARM HEK116 model.

        Args:
            sequences: List of DNA sequences (should be ~230bp for optimal results)

        Returns:
            Array of predicted activity scores
        """
        if not sequences:
            return np.array([])

        # Write sequences to temporary FASTA file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta', delete=False
        ) as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")
            fasta_path = f.name

        output_path = fasta_path.replace('.fasta', '_predictions.txt')

        try:
            # Build PARM command
            cmd = [
                "parm", "predict",
                "--input", fasta_path,
                "--output", output_path,
                "--n_seqs_per_batch", str(self.batch_size),
            ]

            if self.model_path:
                cmd.extend(["--model", self.model_path])
            else:
                # Use HEK116 model (default location after PARM install)
                cmd.extend(["--model", "HEK116"])

            # Run prediction
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"PARM prediction failed: {result.stderr}")

            # Parse predictions
            scores = []
            with open(output_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        # Score is the last column
                        try:
                            score = float(parts[-1])
                            scores.append(score)
                        except ValueError:
                            continue  # Skip header or malformed lines

            return np.array(scores)

        finally:
            # Cleanup temporary files
            Path(fasta_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def score_single(self, sequence: str) -> float:
        """Score a single DNA sequence."""
        scores = self.score_sequences([sequence])
        if len(scores) == 0:
            raise ValueError("Failed to score sequence")
        return float(scores[0])

    def __call__(self, sequences):
        """Make oracle callable like PyTorch models."""
        if isinstance(sequences, str):
            return self.score_single(sequences)
        return self.score_sequences(sequences)


class PARMOracleTorch:
    """
    PARM oracle with PyTorch-like interface for integration with Ctrl-DNA.

    Provides the same interface as EnformerModel for drop-in replacement.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.oracle = PARMOracle(model_path=model_path)
        self.device = "cpu"  # PARM handles its own device

    def to(self, device):
        """No-op for compatibility."""
        return self

    def cuda(self):
        """No-op for compatibility."""
        return self

    def eval(self):
        """No-op for compatibility."""
        return self

    def __call__(self, sequences, return_logits=False):
        """
        Score sequences with PARM.

        Args:
            sequences: DNA sequences (list of strings or tensor)
            return_logits: Ignored (for EnformerModel compatibility)

        Returns:
            Tensor-like array of scores
        """
        import torch

        # Handle different input formats
        if isinstance(sequences, str):
            sequences = [sequences]
        elif hasattr(sequences, 'tolist'):
            # Assume one-hot encoded tensor - need to decode
            # This would require decoding logic
            raise NotImplementedError(
                "PARMOracleTorch expects string sequences, not tensors"
            )

        scores = self.oracle.score_sequences(sequences)
        return torch.tensor(scores).unsqueeze(-1)  # Shape: (N, 1)


def create_hek293_oracle(model_path: Optional[str] = None) -> PARMOracleTorch:
    """
    Create HEK293-like oracle using PARM's HEK116 model.

    This is the recommended function to create an HEK293 OFF-target oracle
    for the tissue-specific promoter optimization pipeline.

    Args:
        model_path: Optional path to PARM HEK116 model directory

    Returns:
        PARMOracleTorch oracle ready for use in optimization
    """
    return PARMOracleTorch(model_path=model_path)
