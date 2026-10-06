import os
import math
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Extracts raw token log-probabilities for binary safety decisions
    and applies validation-calibrated Temperature / Platt scaling.
    
    Formula:
        z = log p("true") - log p("false")
        P_raw = σ(z) = 1 / (1 + exp(-z))
        P_calibrated = σ(z / T + b)
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        bias: Optional[float] = None,
    ):
        """
        Initialize the calibrator.
        
        Args:
            temperature: Temperature parameter T (default from env CONFIDENCE_CALIBRATION_TEMPERATURE or 1.35)
            bias: Platt scaling bias parameter b (default from env CONFIDENCE_CALIBRATION_BIAS or 0.0)
        """
        env_t = float(os.getenv("CONFIDENCE_CALIBRATION_TEMPERATURE", "1.35"))
        env_b = float(os.getenv("CONFIDENCE_CALIBRATION_BIAS", "0.0"))
        
        self.temperature = temperature if temperature is not None else env_t
        self.bias = bias if bias is not None else env_b
        
        # Guard against zero or negative temperature
        if self.temperature <= 0:
            self.temperature = 1.0

    def extract_and_calibrate(
        self,
        result_dict: Dict[str, Any],
        fallback_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extract token logprobs from the LLM result and compute calibrated confidence.
        
        Args:
            result_dict: The dictionary returned by SafetyLLM.invoke()
            fallback_confidence: Fallback float if logprob is unavailable
            
        Returns:
            Dict with:
                - confidence: float (calibrated mathematical probability, 0.0 - 1.0)
                - raw_confidence: float (uncalibrated softmax probability, 0.0 - 1.0)
                - logit_z: Optional[float] (raw log-odds logit z)
                - calibration_source: str (e.g. "logprob_calibrated", "prompt_fallback")
        """
        # Check for logprobs in _logprobs or _raw_response
        logprobs_data = result_dict.get("_logprobs")
        if not logprobs_data and "_raw_response" in result_dict:
            choices = result_dict["_raw_response"].get("choices", [])
            if choices and "logprobs" in choices[0]:
                logprobs_data = choices[0]["logprobs"]

        if not logprobs_data:
            # Fallback to prompt-provided confidence or default
            fb = fallback_confidence if fallback_confidence is not None else float(result_dict.get("confidence", 0.5))
            fb = max(0.0, min(1.0, fb))
            return {
                "confidence": round(fb, 4),
                "raw_confidence": round(fb, 4),
                "logit_z": None,
                "calibration_source": "prompt_fallback",
            }

        # Extract content tokens list
        content_tokens: List[Dict[str, Any]] = []
        if isinstance(logprobs_data, dict):
            content_tokens = logprobs_data.get("content", []) or []
        elif isinstance(logprobs_data, list):
            content_tokens = logprobs_data

        if not content_tokens:
            fb = float(result_dict.get("confidence", 0.5))
            return {
                "confidence": round(fb, 4),
                "raw_confidence": round(fb, 4),
                "logit_z": None,
                "calibration_source": "empty_tokens_fallback",
            }

        # Locate the decision token right after "pred_is_harmful" or "is_harmful"
        target_token_data = None
        for idx, item in enumerate(content_tokens):
            token_str = str(item.get("token", "")).strip().lower().replace('"', '').replace("'", "")
            if "pred_is_harmful" in token_str or "is_harmful" in token_str or "harmful" in token_str:
                # Search up to next 5 tokens for true/false/yes/no
                for next_idx in range(idx + 1, min(idx + 6, len(content_tokens))):
                    candidate = content_tokens[next_idx]
                    cand_str = str(candidate.get("token", "")).strip().lower().replace('"', '').replace("'", "")
                    if cand_str in ("true", "false", "yes", "no"):
                        target_token_data = candidate
                        break
                if target_token_data is not None:
                    break

        if target_token_data is None:
            # If marker wasn't found, check if first boolean token can be located
            for item in content_tokens[:15]:
                tok_str = str(item.get("token", "")).strip().lower().replace('"', '').replace("'", "")
                if tok_str in ("true", "false", "yes", "no"):
                    target_token_data = item
                    break

        if target_token_data is None:
            fb = float(result_dict.get("confidence", 0.5))
            return {
                "confidence": round(fb, 4),
                "raw_confidence": round(fb, 4),
                "logit_z": None,
                "calibration_source": "token_unlocated_fallback",
            }

        # Inspect top_logprobs
        top_logprobs = target_token_data.get("top_logprobs", [])
        chosen_token = str(target_token_data.get("token", "")).strip().lower()
        chosen_logprob = float(target_token_data.get("logprob", -100.0))

        logp_true = -100.0
        logp_false = -100.0

        # Include the chosen token
        if chosen_token in ("true", "yes", ": true", ":true"):
            logp_true = max(logp_true, chosen_logprob)
        elif chosen_token in ("false", "no", ": false", ":false"):
            logp_false = max(logp_false, chosen_logprob)

        # Iterate over alternative candidates in top_logprobs
        for entry in top_logprobs:
            tok = str(entry.get("token", "")).strip().lower().replace('"', '').replace("'", "")
            lp = float(entry.get("logprob", -100.0))
            if tok in ("true", "yes"):
                logp_true = max(logp_true, lp)
            elif tok in ("false", "no"):
                logp_false = max(logp_false, lp)

        # If one class logprob is missing from top-k, compute complement probability safely
        if logp_true > -50 and logp_false <= -50:
            p_true = math.exp(logp_true)
            p_false = max(1e-7, 1.0 - p_true)
            logp_false = math.log(p_false)
        elif logp_false > -50 and logp_true <= -50:
            p_false = math.exp(logp_false)
            p_true = max(1e-7, 1.0 - p_false)
            logp_true = math.log(p_true)
        elif logp_true <= -50 and logp_false <= -50:
            # Neither was found in top-k, fallback to chosen token direction
            if "true" in chosen_token or "yes" in chosen_token:
                logp_true = -0.05
                logp_false = -3.0
            else:
                logp_true = -3.0
                logp_false = -0.05

        # Compute log-odds logit z
        z = logp_true - logp_false

        # Compute uncalibrated softmax probability
        try:
            raw_prob = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            raw_prob = 1.0 if z > 0 else 0.0

        # Compute temperature-scaled calibrated probability
        try:
            z_calibrated = (z / self.temperature) + self.bias
            calibrated_prob = 1.0 / (1.0 + math.exp(-z_calibrated))
        except OverflowError:
            calibrated_prob = 1.0 if z > 0 else 0.0

        return {
            "confidence": round(calibrated_prob, 4),
            "raw_confidence": round(raw_prob, 4),
            "logit_z": round(z, 4),
            "temperature_used": self.temperature,
            "bias_used": self.bias,
            "calibration_source": "logprob_calibrated",
        }
