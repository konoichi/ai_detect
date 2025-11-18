"""
===========================================================
 KI-DETECTOR – IMAGE HEURISTICS MODULE
 Version: v0.5
 Author: Stephan (Projekt: ai_detect)
 Purpose:
   - Verbessert KI-Erkennung mit aggressiveren Schwellenwerten
   - KI-Skin-Analyse (Hautuniformität / Rotkanal-Glättung)
   - Farbharmonie-/Color-Grading-Erkennung
   - Oversharpening-Detection verstärkt
   - KI-Rauschsimulation robust erkannt
   - Papier-/Scan-Entlastung verbessert
===========================================================
"""

import io
import numpy as np
from PIL import (
    Image,
    ExifTags,
    ImageStat,
    ImageOps,
    ImageFilter
)
from typing import Dict, List
from ..utils.logging import logger


# -----------------------------------------------------------
# SELFTEST
# -----------------------------------------------------------
def selftest() -> Dict:
    """Rudimentärer Modul-Selbsttest."""
    try:
        test_img = Image.new("RGB", (400, 400), color="gray")
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        analyze_image(buf.getvalue())
        return {"status": "ok", "version": "v0.5", "module": "image_detector"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


# -----------------------------------------------------------
# HILFSFUNKTIONEN
# -----------------------------------------------------------

def load_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.error(f"Bild konnte nicht geladen werden: {e}")
        raise


# -----------------------------------------------------------
# HEAVY HEURISTICS v0.5
# -----------------------------------------------------------

def detect_paper_or_scan(img: Image.Image, warnings: List[str]) -> float:
    """Entlastet typische Papier-/Scanbilder."""
    gray = img.convert("L")
    pixels = np.array(gray)
    white_ratio = (pixels > 240).mean()

    if white_ratio > 0.65:
        warnings.append("Papier/Scan erkannt – geringes KI-Risiko.")
        return -0.25

    return 0.0


def base_score_from_image(img: Image.Image, warnings: List[str]) -> float:
    """Dynamischer Startwert abhängig von Stil / Licht / Format."""
    base = 0.30

    # Papier/Scan senkt
    base += detect_paper_or_scan(img, warnings)

    w, h = img.size

    # Quadratisch → KI-typisch
    if w == h:
        base += 0.20
        warnings.append("Quadratisches Bildformat – KI-typisch.")

    # Cinematic Lighting (dunkle low-percentiles = künstliche Studiobeleuchtung)
    gray = img.convert("L")
    low_p = np.percentile(np.array(gray), 10)
    if low_p < 40:
        base += 0.15
        warnings.append("Cinematic Lighting erkannt – KI-typischer Stil.")

    return max(0.05, min(base, 0.70))


def exif_score(img: Image.Image, warnings: List[str]) -> float:
    """EXIF stark gewichtet."""
    try:
        exif = img.getexif()
        if not exif or len(exif.keys()) == 0:
            warnings.append("Keine EXIF-Daten gefunden – KI typisch.")
            return 0.10

        # Echte Kamera entlastet stark
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag, tag) == "Make":
                val = str(value).lower()
                if any(k in val for k in ["canon", "sony", "nikon", "fujifilm"]):
                    warnings.append(f"Echte Kamera erkannt ({value}).")
                    return -0.30

    except:
        warnings.append("EXIF nicht lesbar – wirkt KI-typisch.")
        return 0.05

    return 0.0


def oversharp_score(img: Image.Image, warnings: List[str]) -> float:
    """Oversharpening deutlich stärker gewichtet."""
    laplace = img.filter(ImageFilter.FIND_EDGES)
    edge_mean = ImageStat.Stat(laplace).mean[0]

    # Aggressivere Schwelle v0.5
    if edge_mean > 45:
        warnings.append("Übermäßig scharfe Konturen – KI-Oversharpening.")
        return 0.30

    return 0.0


def smoothness_score(img: Image.Image, warnings: List[str]) -> float:
    """KI-Skin – Hautuniformität & Glättung."""
    gray = img.convert("L")
    var = ImageStat.Stat(gray).var[0]

    # Zu glatte Textur → KI
    if var < 260:
        warnings.append("Sehr glatte Haut- / Bildtexturen – KI-Glättung.")
        return 0.25

    # KI-Rauschsymmetrie (gleichmäßiges fake-noise)
    if 260 <= var <= 500:
        warnings.append("Gleichmäßiges Rauschmuster – KI-Rauschsimulation möglich.")
        return 0.15

    return 0.0


def color_score(img: Image.Image, warnings: List[str]) -> float:
    """Color-Grading / KI-Farbharmonie."""
    stat = ImageStat.Stat(img)
    std_r, std_g, std_b = stat.stddev

    saturation = (std_r + std_g + std_b) / 3

    # KI-Farben
    if saturation > 55:
        warnings.append("Starke Farbsättigung – KI-Farbprofil.")
        return 0.20

    # Harmonie – Kanäle zu ähnlich → KI
    if abs(std_r - std_g) < 6 and abs(std_g - std_b) < 6:
        warnings.append("Unnatürlich gleichmäßige Farbharmonie – KI-Hinweis.")
        return 0.15

    return 0.0


def weird_hand_score(img: Image.Image, warnings: List[str]) -> float:
    """Kontur-basierte Artefakt-Erkennung."""
    try:
        small = img.resize((128, 128))
        edges = small.filter(ImageFilter.FIND_EDGES)
        mean_edges = ImageStat.Stat(edges).mean[0]
        if mean_edges > 45:
            warnings.append("Stark segmentierte Konturen – KI-Artefakte möglich.")
            return 0.20
    except:
        pass
    return 0.0


def resolution_score(img: Image.Image, warnings: List[str]) -> float:
    """Auflösung / Format."""
    w, h = img.size
    score = 0.0

    if w in (512, 768, 1024, 1536, 2048):
        warnings.append("Typische KI-Auflösungsgröße erkannt.")
        score += 0.20

    if w * h < 200_000:
        warnings.append("Geringe Auflösung – wirkt wie Scan/Echtbild.")
        score -= 0.15

    return score


# -----------------------------------------------------------
# HAUPTFUNKTION
# -----------------------------------------------------------

def analyze_image(data: bytes) -> Dict:
    img = load_image(data)
    warnings = []

    # Dynamischer BaseScore v0.5
    score = base_score_from_image(img, warnings)

    # Heuristiken (pos/neg)
    score += exif_score(img, warnings)
    score += smoothness_score(img, warnings)
    score += oversharp_score(img, warnings)
    score += color_score(img, warnings)
    score += resolution_score(img, warnings)
    score += weird_hand_score(img, warnings)

    # Score clamp + runden
    score = max(0.0, min(score, 1.0))
    score = round(score, 2)

    logger.info(f"Analyse v0.5: Score={score}, Warnings={len(warnings)}")

    return {
        "is_ai_probability": score,
        "warnings": warnings,
        "dimensions": {"width": img.size[0], "height": img.size[1]},
    }
