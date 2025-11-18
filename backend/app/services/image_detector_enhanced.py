"""
===========================================================
 KI-DETECTOR – IMAGE HEURISTICS PRE-FILTER
 Version: v0.6
 Author: Enhanced by Claude
 Purpose:
   - Fast pre-filter using reliable heuristics
   - Catches obvious AI artifacts only
   - Prepares for ML model integration
   - Conservative scoring to avoid false positives
===========================================================
"""

import io
import hashlib
import numpy as np
from PIL import Image, ExifTags, ImageStat, ImageFilter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

class DetectionLevel(Enum):
    """Pre-filter confidence levels"""
    OBVIOUS_AI = "obvious_ai"          # Very high confidence
    SUSPICIOUS = "suspicious"           # Moderate confidence - needs ML
    UNCERTAIN = "uncertain"             # Low confidence - needs ML
    LIKELY_REAL = "likely_real"        # High confidence real photo


@dataclass
class HeuristicConfig:
    """Configurable thresholds for heuristics"""
    # File size limits
    max_file_size_mb: int = 20
    max_dimension: int = 8192
    min_dimension: int = 32
    
    # AI artifact thresholds (conservative)
    extreme_sharpness_threshold: float = 70.0  # Very high = obvious oversharpening
    extreme_smoothness_threshold: float = 150.0  # Very low variance = obvious smoothing
    
    # Format red flags
    known_ai_resolutions: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.known_ai_resolutions is None:
            # Common default AI generator resolutions
            self.known_ai_resolutions = [
                (512, 512), (768, 768), (1024, 1024),
                (512, 768), (768, 512),
                (1024, 1536), (1536, 1024)
            ]


CONFIG = HeuristicConfig()


# -----------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------

class ImageValidationError(Exception):
    """Raised when image fails validation"""
    pass


def validate_image_data(data: bytes) -> None:
    """Validate image data before processing"""
    if not data:
        raise ImageValidationError("Empty image data")
    
    size_mb = len(data) / (1024 * 1024)
    if size_mb > CONFIG.max_file_size_mb:
        raise ImageValidationError(
            f"Image too large: {size_mb:.1f}MB (max: {CONFIG.max_file_size_mb}MB)"
        )
    
    # Check if it's actually an image
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        
        if w > CONFIG.max_dimension or h > CONFIG.max_dimension:
            raise ImageValidationError(
                f"Dimensions too large: {w}x{h} (max: {CONFIG.max_dimension})"
            )
        
        if w < CONFIG.min_dimension or h < CONFIG.min_dimension:
            raise ImageValidationError(
                f"Dimensions too small: {w}x{h} (min: {CONFIG.min_dimension})"
            )
        
        img.close()
    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        raise ImageValidationError(f"Invalid image format: {str(e)}")


def load_image(data: bytes) -> Image.Image:
    """Load and convert image to RGB"""
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise ImageValidationError(f"Failed to load image: {str(e)}")


# -----------------------------------------------------------
# CONSERVATIVE HEURISTICS (Pre-Filter Only)
# -----------------------------------------------------------

def check_obvious_ai_artifacts(img: Image.Image) -> Tuple[float, List[str]]:
    """
    Check for OBVIOUS AI artifacts only.
    Returns: (confidence_score, list_of_reasons)
    Score: 0.0 (no artifacts) to 1.0 (obvious AI)
    """
    score = 0.0
    flags = []
    
    # 1. EXTREME oversharpening (obvious processing)
    try:
        edges = img.filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(edges).mean[0]
        
        if edge_mean > CONFIG.extreme_sharpness_threshold:
            score += 0.30
            flags.append(f"Extreme sharpening detected (edge_mean={edge_mean:.1f})")
    except Exception as e:
        flags.append(f"Edge detection failed: {str(e)}")
    
    # 2. EXTREME smoothness (obvious airbrushing/AI smoothing)
    try:
        gray = img.convert("L")
        variance = ImageStat.Stat(gray).var[0]
        
        if variance < CONFIG.extreme_smoothness_threshold:
            score += 0.25
            flags.append(f"Extreme smoothness detected (variance={variance:.1f})")
    except Exception as e:
        flags.append(f"Smoothness check failed: {str(e)}")
    
    # 3. Check for impossible geometry (downscaled check for speed)
    try:
        small = img.resize((256, 256), Image.Resampling.LANCZOS)
        edges_small = small.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges_small.convert("L"))
        
        # Look for highly regular patterns (AI grid artifacts)
        if edge_arr.std() < 10 and edge_arr.mean() > 50:
            score += 0.20
            flags.append("Regular pattern detected (possible AI grid artifacts)")
    except Exception as e:
        flags.append(f"Geometry check failed: {str(e)}")
    
    return min(score, 1.0), flags


def check_metadata_red_flags(img: Image.Image) -> Tuple[float, List[str]]:
    """
    Check metadata for AI red flags.
    Conservative: Only flag obvious AI markers.
    """
    score = 0.0
    flags = []
    
    try:
        # Check for AI software signatures in EXIF
        exif = img.getexif()
        if exif:
            software_tags = []
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name in ["Software", "ProcessingSoftware", "Model"]:
                    software_tags.append(str(value).lower())
            
            # Known AI generators
            ai_markers = ["midjourney", "dalle", "stable diffusion", "stablediffusion", 
                         "dreamstudio", "leonardo", "firefly"]
            
            for tag_value in software_tags:
                for marker in ai_markers:
                    if marker in tag_value:
                        score += 0.50
                        flags.append(f"AI software detected in metadata: {marker}")
                        break
    
    except Exception as e:
        # EXIF errors are not suspicious - just skip
        pass
    
    return min(score, 1.0), flags


def check_format_patterns(img: Image.Image) -> Tuple[float, List[str]]:
    """
    Check image format for AI patterns.
    Only flag exact matches to known AI resolutions.
    """
    score = 0.0
    flags = []
    
    w, h = img.size
    
    # Exact match to common AI generator defaults
    if (w, h) in CONFIG.known_ai_resolutions or (h, w) in CONFIG.known_ai_resolutions:
        score += 0.15
        flags.append(f"Exact match to common AI resolution: {w}x{h}")
    
    return score, flags


def check_color_anomalies(img: Image.Image) -> Tuple[float, List[str]]:
    """
    Check for AI-typical color processing.
    Conservative: Only extreme cases.
    """
    score = 0.0
    flags = []
    
    try:
        stat = ImageStat.Stat(img)
        std_r, std_g, std_b = stat.stddev
        
        # EXTREME color uniformity (all channels nearly identical)
        max_diff = max(abs(std_r - std_g), abs(std_g - std_b), abs(std_r - std_b))
        if max_diff < 3.0:
            score += 0.15
            flags.append(f"Extreme color uniformity detected (max_diff={max_diff:.2f})")
        
        # EXTREME saturation (unrealistic color boost)
        avg_std = (std_r + std_g + std_b) / 3
        if avg_std > 70:
            score += 0.15
            flags.append(f"Extreme color saturation detected (avg_std={avg_std:.1f})")
    
    except Exception as e:
        flags.append(f"Color analysis failed: {str(e)}")
    
    return score, flags


# -----------------------------------------------------------
# MAIN PRE-FILTER FUNCTION
# -----------------------------------------------------------

def prefilter_heuristics(img: Image.Image) -> Dict:
    """
    Run conservative heuristics as pre-filter.
    Returns structured results for decision making.
    """
    all_flags = []
    total_score = 0.0
    
    # Run all heuristic checks
    checks = [
        ("artifacts", check_obvious_ai_artifacts),
        ("metadata", check_metadata_red_flags),
        ("format", check_format_patterns),
        ("color", check_color_anomalies),
    ]
    
    scores_breakdown = {}
    
    for check_name, check_func in checks:
        try:
            score, flags = check_func(img)
            total_score += score
            all_flags.extend(flags)
            scores_breakdown[check_name] = {
                "score": round(score, 3),
                "flags": flags
            }
        except Exception as e:
            all_flags.append(f"{check_name} check failed: {str(e)}")
            scores_breakdown[check_name] = {
                "score": 0.0,
                "flags": [f"Check failed: {str(e)}"]
            }
    
    # Clamp total score
    total_score = max(0.0, min(total_score, 1.0))
    
    # Determine detection level
    if total_score >= 0.70:
        level = DetectionLevel.OBVIOUS_AI
        needs_ml = False
    elif total_score >= 0.40:
        level = DetectionLevel.SUSPICIOUS
        needs_ml = True
    elif total_score >= 0.20:
        level = DetectionLevel.UNCERTAIN
        needs_ml = True
    else:
        level = DetectionLevel.LIKELY_REAL
        needs_ml = False
    
    return {
        "prefilter_score": round(total_score, 3),
        "detection_level": level.value,
        "needs_ml_verification": needs_ml,
        "flags": all_flags,
        "scores_breakdown": scores_breakdown,
    }


def analyze_image(data: bytes) -> Dict:
    """
    Main entry point for image analysis.
    Validates input and runs pre-filter heuristics.
    """
    try:
        # Validate input
        validate_image_data(data)
        
        # Load image
        img = load_image(data)
        w, h = img.size
        
        # Calculate image hash for caching/deduplication
        img_hash = hashlib.sha256(data).hexdigest()[:16]
        
        # Run pre-filter heuristics
        heuristic_results = prefilter_heuristics(img)
        
        # Build response
        result = {
            "status": "success",
            "image_hash": img_hash,
            "dimensions": {"width": w, "height": h},
            "file_size_kb": round(len(data) / 1024, 2),
            **heuristic_results,
            "ml_required": heuristic_results["needs_ml_verification"],
            "recommendation": _get_recommendation(heuristic_results),
        }
        
        img.close()
        return result
    
    except ImageValidationError as e:
        return {
            "status": "error",
            "error_type": "validation_error",
            "error_message": str(e),
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_type": "processing_error",
            "error_message": str(e),
        }


def _get_recommendation(heuristic_results: Dict) -> str:
    """Generate human-readable recommendation"""
    level = heuristic_results["detection_level"]
    
    if level == DetectionLevel.OBVIOUS_AI.value:
        return "High confidence AI detection - multiple obvious artifacts found"
    elif level == DetectionLevel.SUSPICIOUS.value:
        return "Suspicious patterns detected - ML verification recommended"
    elif level == DetectionLevel.UNCERTAIN.value:
        return "Inconclusive - ML analysis required for accurate detection"
    else:
        return "No obvious AI artifacts detected - likely authentic photo"


# -----------------------------------------------------------
# SELFTEST
# -----------------------------------------------------------

def selftest() -> Dict:
    """Module self-test with multiple scenarios"""
    results = []
    
    # Test 1: Basic functionality
    try:
        test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG", quality=95)
        result = analyze_image(buf.getvalue())
        results.append({
            "test": "basic_functionality",
            "status": "pass" if result["status"] == "success" else "fail",
            "details": result.get("error_message", "OK")
        })
    except Exception as e:
        results.append({
            "test": "basic_functionality",
            "status": "fail",
            "details": str(e)
        })
    
    # Test 2: Validation (oversized file)
    try:
        large_data = b"x" * (CONFIG.max_file_size_mb * 1024 * 1024 + 1)
        result = analyze_image(large_data)
        results.append({
            "test": "size_validation",
            "status": "pass" if result["status"] == "error" else "fail",
            "details": result.get("error_message", "Should have failed")
        })
    except Exception as e:
        results.append({
            "test": "size_validation",
            "status": "fail",
            "details": str(e)
        })
    
    # Test 3: Invalid image data
    try:
        result = analyze_image(b"not an image")
        results.append({
            "test": "invalid_data",
            "status": "pass" if result["status"] == "error" else "fail",
            "details": result.get("error_message", "Should have failed")
        })
    except Exception as e:
        results.append({
            "test": "invalid_data",
            "status": "fail",
            "details": str(e)
        })
    
    all_passed = all(r["status"] == "pass" for r in results)
    
    return {
        "status": "ok" if all_passed else "error",
        "version": "v0.6",
        "module": "image_detector",
        "config": {
            "max_file_size_mb": CONFIG.max_file_size_mb,
            "max_dimension": CONFIG.max_dimension,
        },
        "tests": results,
        "summary": f"{sum(1 for r in results if r['status'] == 'pass')}/{len(results)} tests passed"
    }


# -----------------------------------------------------------
# ML INTEGRATION STUB (for next step)
# -----------------------------------------------------------

def analyze_with_ml(data: bytes, prefilter_results: Dict) -> Dict:
    """
    Placeholder for ML model integration.
    Will use Hugging Face transformers model for AI detection.
    """
    raise NotImplementedError(
        "ML analysis not yet implemented. "
        "Next step: integrate Hugging Face AI detection model"
    )
