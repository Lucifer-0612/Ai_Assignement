"""
PhytoScan AI — Core Diagnostic & Computer Vision Engine
======================================================
Dual-tier hybrid plant leaf disease prediction and diagnostic engine:
- Tier 1: Optional HuggingFace MobileNetV2 Vision Transformer/CNN pipeline
- Tier 2: Pure-Python CV Color-Space Feature Extraction + Random Forest Classifier (100% Offline)
- Procedural Sample Leaf Generator for offline demonstration
- Comprehensive Agronomic Pathology Knowledge Base & Treatment Prescriptions
"""

from __future__ import annotations

import os
import time
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.ensemble import RandomForestClassifier

# Set up module logger
logger = logging.getLogger("PhytoScan.Engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================================
# 1. DATA CONTRACTS
# ============================================================================

@dataclass
class DiseaseProfile:
    """Represents the complete pathological, agronomic, and treatment profile of a plant disease."""
    disease_id: str
    name: str
    pathogen_type: str  # "Fungal", "Bacterial", "Viral", or "None (Healthy)"
    symptoms: str
    organic_treatment: List[str]
    chemical_treatment: List[str]
    prevention_tips: List[str]


@dataclass
class PredictionResult:
    """Encapsulates comprehensive diagnostic output from the computer vision and ML pipeline."""
    predicted_disease: DiseaseProfile
    confidence_pct: float
    top_3_predictions: List[Tuple[str, float]]
    infected_area_pct: float
    severity_level: str  # "None (Healthy)", "Mild (1–10%)", "Moderate (11–30%)", "Severe (>30%)"
    feature_metrics: Dict[str, float]
    original_image: Image.Image
    segmented_overlay_image: Image.Image
    engine_used: str
    processing_time_ms: float


# ============================================================================
# 2. AGRONOMIC PATHOLOGY KNOWLEDGE BASE
# ============================================================================

DISEASE_DATABASE: Dict[str, DiseaseProfile] = {
    "healthy": DiseaseProfile(
        disease_id="healthy",
        name="Healthy Plant Leaf (No Pathogen Detected)",
        pathogen_type="None (Healthy)",
        symptoms=(
            "Uniform vibrant emerald/forest green leaf lamina with intact waxy cuticle. "
            "Clean vascular midrib and lateral venation. Absence of chlorosis, necrosis, "
            "blistering, or fungal sporulation."
        ),
        organic_treatment=[
            "Apply balanced compost tea or seaweed foliar extract bi-weekly to sustain plant vigor.",
            "Maintain soil biodiversity with beneficial mycorrhizal fungi and Trichoderma soil inoculants.",
            "Ensure regular deep drip watering at the root zone during cool early morning hours.",
            "Apply cold-pressed neem oil (0.5% emulsion) every 14 days as a preventive barrier against insect vectors."
        ],
        chemical_treatment=[
            "No chemical intervention or pesticide application required.",
            "Preserve beneficial predator insects (ladybugs, lacewings) and soil microbial ecology.",
            "Conduct seasonal soil mineral testing to maintain optimal N-P-K and micronutrient balance."
        ],
        prevention_tips=[
            "Sterilize pruning shears and garden implements with 70% isopropyl alcohol between plants.",
            "Avoid overhead sprinkler irrigation to keep foliage dry and minimize spore germination risk.",
            "Ensure recommended intra-row crop spacing to encourage laminar canopy airflow.",
            "Implement a strict 3- to 4-year crop rotation schedule across different botanical families."
        ],
    ),
    "cercospora_beet": DiseaseProfile(
        disease_id="cercospora_beet",
        name="Sugar Beet (Beta vulgaris) - Cercospora Leaf Spot (Cercospora beticola)",
        pathogen_type="Fungal",
        symptoms=(
            "Numerous small, circular to sub-circular lesions (2-5 mm) with tan to ash-gray necrotic "
            "centers and distinct dark reddish-brown to dark purple margins. Lesions are bordered by "
            "prominent chlorotic yellow halos. Spots coalesce rapidly under warm, humid conditions "
            "(>25°C and >90% RH), causing extensive foliar blighting and premature crown senescence."
        ),
        organic_treatment=[
            "Apply liquid fixed copper hydroxide or copper octanoate (copper soap) at first visible spot detection.",
            "Foliar bio-fungicide applications containing Bacillus subtilis (QST 713) or Bacillus amyloliquefaciens.",
            "Apply systemic acquired resistance (SAR) inducers including potassium phosphite and soluble silicon.",
            "Prune and deeply incorporate infected crop residues immediately post-harvest to halt spore carryover."
        ],
        chemical_treatment=[
            "Apply protective multi-site contact fungicides: Mancozeb, Triphenyltin hydroxide (TPTH), or Chlorothalonil.",
            "Apply systemic triazoles (FRAC Group 3: Difenoconazole, Prothioconazole, or Tetraconazole) tank-mixed with multi-site contacts.",
            "Rotate FRAC Group 11 strobilurins (Azoxystrobin or Pyraclostrobin) strictly adhering to resistance management guidelines."
        ],
        prevention_tips=[
            "Plant certified Cercospora-tolerant / resistant sugar beet hybrid seed varieties (CR+ genetics).",
            "Enforce a mandatory 3-year crop rotation away from Chenopodiaceae (sugar beet, table beet, spinach).",
            "Maintain minimum 100-meter spatial isolation from previous year's infested sugar beet fields.",
            "Avoid late-afternoon overhead irrigation; utilize precision furrow or drip systems to reduce canopy leaf wetness duration."
        ],
    ),
    "early_blight": DiseaseProfile(
        disease_id="early_blight",
        name="Tomato / Potato - Early Blight (Alternaria solani)",
        pathogen_type="Fungal",
        symptoms=(
            "Characterized by distinct dark brown to black necrotic spots with concentric "
            "conidial rings resembling a target board or tree rings. Lesions are typically "
            "bordered by a bright chlorotic yellow halo and initially appear on older, lower leaves."
        ),
        organic_treatment=[
            "Prune off infected lower leaves up to 12-18 inches above the soil line and dispose off-site in sealed bags.",
            "Apply liquid copper octanoate (copper soap) or Bordeaux mixture at 7- to 10-day intervals.",
            "Spray bio-fungicides containing Bacillus subtilis (e.g., Serenade) or Bacillus amyloliquefaciens.",
            "Apply potassium bicarbonate (3 tbsp/gal with insecticidal spreader-sticker) to alter leaf surface pH."
        ],
        chemical_treatment=[
            "Apply protective broad-spectrum contact fungicides: Chlorothalonil or Mancozeb before canopy closure.",
            "For curative systemic control during active outbreaks, alternate FRAC group 11 (Azoxystrobin/Pyraclostrobin) with FRAC group 3 (Difenoconazole).",
            "Follow strict resistance management by rotating chemical modes of action every two sprays."
        ],
        prevention_tips=[
            "Apply 2-3 inches of straw, woodchips, or black plastic mulch to prevent rain-splash soil inoculation.",
            "Utilize drip or soaker hose irrigation exclusively; never wet foliage late in the afternoon.",
            "Select certified disease-resistant or tolerant cultivars (look for 'EB' designation on seed packets).",
            "Burn or deeply bury all solanaceous crop residues immediately following post-season harvest."
        ],
    ),
    "powdery_mildew": DiseaseProfile(
        disease_id="powdery_mildew",
        name="Cucurbit / Squash - Powdery Mildew (Podosphaera xanthii)",
        pathogen_type="Fungal",
        symptoms=(
            "Superficial white to light-gray powdery or talcum-like mycelial patches developing on "
            "the upper and lower leaf surfaces, petioles, and stems. Severely infected leaves turn "
            "chlorotic, curl upward, brown prematurely, and undergo rapid senescence."
        ),
        organic_treatment=[
            "Foliar spray with potassium bicarbonate (30g per 10 liters water) blended with horticultural mineral oil.",
            "Fresh milk foliar solution (40% whole cow milk, 60% water) applied in direct midday sunlight to generate natural antiseptic free radicals.",
            "Apply cold-pressed neem oil (1% concentration with mild surfactant) to dissolve active fungal conidia.",
            "Sulfur dust or wettable sulfur suspension applied preventively during warm, dry weather."
        ],
        chemical_treatment=[
            "Apply sterol biosynthesis inhibitor (DMI) systemic fungicides such as Myclobutanil or Triadimefon.",
            "Alternate with QoI strobilurins (Trifloxystrobin or Kresoxim-methyl) or Quinoxyfen.",
            "Avoid sulfur applications when ambient air temperatures exceed 32°C (90°F) to prevent phytotoxic leaf scald."
        ],
        prevention_tips=[
            "Plant exclusively in full direct sunlight (minimum 6-8 hours daily exposure).",
            "Maximize plant spacing and stake/trellis vines vertically to eliminate trapped canopy humidity.",
            "Choose resistant F1 hybrid cultivars bred for powdery mildew immunity.",
            "Avoid excessive synthetic high-nitrogen fertilization which promotes tender, highly susceptible succulent tissue."
        ],
    ),
    "yellow_leaf_curl": DiseaseProfile(
        disease_id="yellow_leaf_curl",
        name="Tomato - Yellow Leaf Curl Virus (TYLCV)",
        pathogen_type="Viral",
        symptoms=(
            "Severe interveinal chlorosis (yellowing) with green vein retention, pronounced upward "
            "cupping and curling of leaflet margins, severe leaf size reduction, stunting of the "
            "apical canopy, and total floral abortion."
        ),
        organic_treatment=[
            "There is no chemical or biological cure for viral plant infections; focus strictly on vector control.",
            "Deploy bright yellow sticky card traps (1 trap per 20 sq. meters) to monitor and capture whitefly vectors (Bemisia tabaci).",
            "Spray botanical pyrethrins or neem-based azadirachtin directly onto the undersides of leaves where whiteflies colonize.",
            "Immediately rogue (uproot) and incinerate symptomatic plants inside plastic bags to prevent vector acquisition."
        ],
        chemical_treatment=[
            "Apply systemic neonicotinoids (Imidacloprid, Thiamethoxam, or Dinotefuran) via soil drench at transplanting.",
            "Rotate foliar sprays with insect growth regulators (Pyriproxyfen or Buprofezin) and lipid synthesis inhibitors (Spiromesifen).",
            "Apply Diamides (Cyantraniliprole) targeting both adult and nymph whitefly stages."
        ],
        prevention_tips=[
            "Cover greenhouse vents and open-field nursery beds with 50-mesh fine insect exclusion netting.",
            "Lay silver or aluminum reflective plastic mulch across planting beds to repel landing whiteflies.",
            "Transplant certified TYLCV-resistant hybrid tomato cultivars carrying Ty-1, Ty-2, or Ty-3 resistance loci.",
            "Maintain a strict 2-month host-free crop break between overlapping solanaceous planting seasons."
        ],
    ),
    "bacterial_spot": DiseaseProfile(
        disease_id="bacterial_spot",
        name="Pepper / Tomato - Bacterial Leaf Spot (Xanthomonas campestris)",
        pathogen_type="Bacterial",
        symptoms=(
            "Numerous small (1-3 mm), angular or circular, water-soaked dark brown to black spots. "
            "Lesions often exhibit translucent margins, coalesce into ragged patches, and tear open "
            "creating a 'shot-hole' appearance. Defoliation occurs rapidly under warm, humid conditions."
        ),
        organic_treatment=[
            "Spray fixed copper hydroxide or copper sulfate combined with Bacillus amyloliquefaciens (strain D747).",
            "Foliar application of botanical bactericides containing thymol, thyme oil, or clove extract.",
            "Apply systemic acquired resistance (SAR) inducers such as harpin protein or soluble potassium silicate.",
            "Immediately prune severely infected branches during dry, sunny weather."
        ],
        chemical_treatment=[
            "Tank-mix copper bactericides with an EBDC fungicide (Mancozeb) to release free copper ions lethal to copper-resistant Xanthomonas strains.",
            "Apply agricultural antibiotic sprays (Streptomycin sulfate or Oxytetracycline) where legally approved for nursery beds.",
            "Treat plants with Acibenzolar-S-methyl (Actigard) to stimulate the plant's endogenous immune defenses."
        ],
        prevention_tips=[
            "Use certified pathogen-free seeds treated with hot water (50°C / 122°F for 25 minutes) or 20% bleach soak.",
            "Strictly avoid working in, cultivating, or harvesting fields when foliage is damp with dew or rain.",
            "Disinfect greenhouse benches, transplant trays, and trellising stakes with a 10% quaternary ammonium or bleach solution.",
            "Rotate out of solanaceous crops (tomatoes, peppers, eggplants) for at least 2 full calendar years."
        ],
    ),
    "late_blight": DiseaseProfile(
        disease_id="late_blight",
        name="Potato / Tomato - Late Blight (Phytophthora infestans)",
        pathogen_type="Fungal",
        symptoms=(
            "Rapidly expanding, irregular, water-soaked pale to dark olive-brown lesions that rapidly "
            "turn purplish-black and necrotic. Under damp/humid conditions, delicate white cottony "
            "fungal sporulation (sporangiophores) appears on the lower leaf surface, followed by total collapse."
        ),
        organic_treatment=[
            "Apply protective Bordeaux mixture (copper sulfate + hydrated lime) thoroughly covering both leaf surfaces.",
            "Foliar inoculation with bio-antagonistic fungi (Trichoderma virens or Trichoderma harzianum).",
            "Immediately destroy, bag, and incinerate all infected plants; do not compost blighted material.",
            "Apply bio-rational potassium phosphite (phosphorous acid salts) to trigger systemic defense signaling."
        ],
        chemical_treatment=[
            "Apply Oomycete-specific systemic translaminar fungicides: Metalaxyl-M / Mefenoxam, Dimethomorph, or Mandipropamid.",
            "Apply kick-back curative fungicides such as Cymoxanil tank-mixed with protective Mancozeb or Chlorothalonil within 24–48 hours of infection.",
            "Use Fluopicolide or Cyazofamid for high disease pressure situations."
        ],
        prevention_tips=[
            "Plant only certified disease-free seed potato tubers and inspect tomato transplants meticulously.",
            "Destroy all volunteer potato plants and post-harvest cull piles before spring emergence.",
            "Track regional BlightCast / late blight epidemiological weather warning systems (cool, wet spells).",
            "Ensure proper soil hilling over potato rows (minimum 10 cm coverage) to prevent tuber blight infection."
        ],
    ),
}


# ============================================================================
# 3. PROCEDURAL SAMPLE LEAF GENERATOR
# ============================================================================

def _draw_leaf_base(size: Tuple[int, int] = (400, 400)) -> Tuple[Image.Image, np.ndarray]:
    """
    Renders an organic, realistic botanical leaf blade on a dark studio slate backdrop.
    Returns the RGB PIL Image and a boolean numpy leaf mask.
    """
    width, height = size
    # Dark studio background with subtle natural grain
    bg_color = (24, 28, 34)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Add slight background noise texture
    bg_noise = np.random.randint(-4, 5, (height, width, 3), dtype=np.int16)
    base_arr = np.array(img, dtype=np.int16) + bg_noise
    base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(img)

    # Parametric leaf contour
    # Central stem runs from (200, 360) down to base, tip at (200, 50)
    top_y = 52
    base_y = 350
    center_x = 200
    leaf_len = base_y - top_y
    max_half_width = 95

    left_points = []
    right_points = []
    y_steps = 140

    for i in range(y_steps + 1):
        fraction = i / y_steps
        y = top_y + fraction * leaf_len
        # Natural ovate-lanceolate curve: sin profile tapering at tip and base
        if fraction < 0.15:
            # Pointed tip
            w = max_half_width * (fraction / 0.15) ** 0.85
        elif fraction < 0.70:
            # Broad lamina
            w = max_half_width * (0.95 + 0.05 * math.sin(fraction * math.pi))
        else:
            # Tapering petiole
            taper = (1.0 - fraction) / 0.30
            w = max_half_width * (taper ** 0.90)

        # Subtle organic irregularity
        w += math.sin(fraction * 18.0) * 2.5 + math.cos(fraction * 35.0) * 1.5
        w = max(1.0, w)

        left_points.append((center_x - w, y))
        right_points.append((center_x + w, y))

    polygon_points = left_points + right_points[::-1]

    # Draw vibrant green base leaf blade with natural gradient
    leaf_mask_img = Image.new("L", (width, height), 0)
    leaf_mask_draw = ImageDraw.Draw(leaf_mask_img)
    leaf_mask_draw.polygon(polygon_points, fill=255)

    # Create base leaf texture
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    # Subtle longitudinal and radial shading
    dist_center = np.abs(x_coords - center_x) / max_half_width
    gradient = 1.0 - 0.22 * dist_center
    gradient = np.clip(gradient, 0.70, 1.08)

    # Base foliage color: Rich Forest Green
    base_r = (38 * gradient).astype(np.uint8)
    base_g = (128 * gradient).astype(np.uint8)
    base_b = (46 * gradient).astype(np.uint8)
    leaf_texture = np.stack([base_r, base_g, base_b], axis=-1)

    # Add cellular microscopic noise
    leaf_noise = np.random.randint(-6, 7, leaf_texture.shape, dtype=np.int16)
    leaf_texture = np.clip(leaf_texture.astype(np.int16) + leaf_noise, 0, 255).astype(np.uint8)

    # Composite leaf onto background using mask
    leaf_mask_arr = np.array(leaf_mask_img) > 128
    composite = np.array(img)
    composite[leaf_mask_arr] = leaf_texture[leaf_mask_arr]
    img = Image.fromarray(composite)
    draw = ImageDraw.Draw(img)

    # Draw Petiole / Stem at base
    stem_color = (68, 142, 62)
    draw.line([(center_x, base_y), (center_x + 2, base_y + 35)], fill=stem_color, width=4)

    # Draw Central Midrib (light lime-green vein)
    vein_color = (95, 175, 82)
    draw.line([(center_x, top_y + 5), (center_x, base_y)], fill=vein_color, width=3)

    # Draw Secondary Lateral Veins
    num_veins = 8
    for v in range(num_veins):
        v_frac = 0.15 + (v / num_veins) * 0.75
        vy_start = top_y + v_frac * leaf_len
        # Veins curve upward toward leaf margins
        curve_drop = 18.0
        # Right lateral
        r_x = center_x + max_half_width * math.sin(v_frac * math.pi) * 0.88
        r_y = vy_start - curve_drop
        draw.line([(center_x, vy_start), (center_x + 15, vy_start - 4), (r_x, r_y)], fill=(85, 162, 75), width=2)
        # Left lateral
        l_x = center_x - max_half_width * math.sin(v_frac * math.pi) * 0.88
        l_y = vy_start - curve_drop
        draw.line([(center_x, vy_start), (center_x - 15, vy_start - 4), (l_x, l_y)], fill=(85, 162, 75), width=2)

    return img, leaf_mask_arr


def generate_sample_leaf_images(output_dir: str = "sample_leaves") -> Dict[str, str]:
    """
    Generates 7 highly realistic procedural synthetic leaf images representing:
    1. Healthy Leaf
    2. Cercospora Leaf Spot (Sugar Beet)
    3. Early Blight (Tomato/Potato)
    4. Powdery Mildew (Squash)
    5. Yellow Leaf Curl Virus (Tomato)
    6. Bacterial Leaf Spot (Pepper)
    7. Late Blight (Potato)

    Images are saved as 400x400 PNGs in output_dir. If already existing, returns paths directly.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "healthy": out_path / "01_healthy_leaf.png",
        "cercospora_beet": out_path / "02_cercospora_sugar_beet.png",
        "early_blight": out_path / "03_early_blight_tomato.png",
        "powdery_mildew": out_path / "04_powdery_mildew_squash.png",
        "yellow_leaf_curl": out_path / "05_yellow_leaf_curl_virus.png",
        "bacterial_spot": out_path / "06_bacterial_leaf_spot_pepper.png",
        "late_blight": out_path / "07_late_blight_potato.png",
    }

    # Verify if all 7 images already exist
    all_exist = all(p.exists() for p in manifest.values())
    if all_exist:
        logger.info(f"Procedural sample leaf library already verified at: {out_path.resolve()}")
        return {k: str(v.resolve()) for k, v in manifest.items()}

    logger.info(f"Synthesizing 7 realistic diagnostic leaf specimens in: {out_path.resolve()}")
    np.random.seed(42)

    # 1. Healthy Leaf
    img_healthy, _ = _draw_leaf_base()
    img_healthy.save(manifest["healthy"], "PNG")

    # 2. Cercospora Leaf Spot (Sugar Beet - small circular spots with chlorotic halo and tan center)
    img_cb, leaf_mask_cb = _draw_leaf_base()
    draw_cb = ImageDraw.Draw(img_cb)
    np.random.seed(202)
    for _ in range(55):
        cx = np.random.randint(130, 270)
        cy = np.random.randint(80, 330)
        if leaf_mask_cb[cy, cx]:
            r = np.random.randint(6, 12)
            # Outer chlorotic yellow halo
            draw_cb.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(205, 192, 42))
            # Reddish-brown border ring
            r_mid = max(2, int(r * 0.65))
            draw_cb.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid], fill=(112, 48, 30))
            # Ash-gray / tan necrotic center
            r_core = max(1, int(r * 0.38))
            draw_cb.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], fill=(170, 160, 150))
    img_cb.save(manifest["cercospora_beet"], "PNG")

    # 3. Early Blight (Concentric dark brown rings with chlorotic yellow halos)
    img_eb, _ = _draw_leaf_base()
    draw_eb = ImageDraw.Draw(img_eb)
    eb_centers = [(175, 140), (230, 195), (160, 240), (220, 275), (185, 310), (145, 180)]
    for cx, cy in eb_centers:
        radius = np.random.randint(18, 30)
        # Outer chlorotic yellow halo
        draw_eb.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(195, 185, 42))
        # Outer dark necrotic ring
        r_mid = int(radius * 0.72)
        draw_eb.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid], fill=(72, 38, 22))
        # Inner lighter ring
        r_inner1 = int(radius * 0.50)
        draw_eb.ellipse([cx - r_inner1, cy - r_inner1, cx + r_inner1, cy + r_inner1], fill=(110, 65, 32))
        # Concentric target core
        r_inner2 = int(radius * 0.28)
        draw_eb.ellipse([cx - r_inner2, cy - r_inner2, cx + r_inner2, cy + r_inner2], fill=(42, 20, 12))
    img_eb.save(manifest["early_blight"], "PNG")

    # 3. Powdery Mildew (White/gray powdery fungal patches)
    img_pm, leaf_mask = _draw_leaf_base()
    pm_arr = np.array(img_pm).astype(np.float32)
    # Scatter 12 powdery colonies with soft gaussian feathering
    pm_centers = [(180, 120), (225, 150), (160, 190), (235, 210), (190, 250), (165, 290),
                  (220, 300), (200, 175), (145, 225), (240, 260), (175, 215), (210, 110)]
    y_grid, x_grid = np.mgrid[0:400, 0:400]
    for cx, cy in pm_centers:
        r_spot = np.random.uniform(20.0, 34.0)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        colony_mask = np.exp(-dist_sq / (2.0 * (r_spot * 0.5) ** 2))
        # Add powdery grain
        powder_noise = np.random.uniform(0.75, 1.15, (400, 400))
        intensity = colony_mask * powder_noise
        intensity = np.clip(intensity, 0.0, 0.90)[:, :, np.newaxis]
        # Blend powdery white/light silver (235, 238, 245)
        powder_color = np.array([238.0, 240.0, 246.0], dtype=np.float32)
        pm_arr = pm_arr * (1.0 - intensity) + powder_color * intensity

    pm_final = np.clip(pm_arr, 0, 255).astype(np.uint8)
    # Mask to leaf boundary
    base_bg = np.array(img_healthy)
    pm_final[~leaf_mask] = base_bg[~leaf_mask]
    Image.fromarray(pm_final).save(manifest["powdery_mildew"], "PNG")

    # 4. Yellow Leaf Curl Virus (Intense chlorosis on leaf margins, pale veins)
    img_ylc, leaf_mask = _draw_leaf_base()
    ylc_arr = np.array(img_ylc).astype(np.float32)
    y_coords, x_coords = np.mgrid[0:400, 0:400]
    # Strong marginal chlorosis: distance from leaf center
    dist_from_midrib = np.abs(x_coords - 200) / 95.0
    chlorosis_intensity = np.clip((dist_from_midrib - 0.28) / 0.65, 0.0, 1.0)
    # Upward curled tips have higher chlorosis
    top_factor = np.clip((250 - y_coords) / 200.0, 0.0, 0.5)
    total_chlorosis = np.clip(chlorosis_intensity + top_factor, 0.0, 0.92)[:, :, np.newaxis]

    # Vivid chlorotic yellow color (228, 215, 38)
    yellow_color = np.array([225.0, 212.0, 36.0], dtype=np.float32)
    ylc_arr = ylc_arr * (1.0 - total_chlorosis) + yellow_color * total_chlorosis
    ylc_final = np.clip(ylc_arr, 0, 255).astype(np.uint8)
    ylc_final[~leaf_mask] = base_bg[~leaf_mask]
    Image.fromarray(ylc_final).save(manifest["yellow_leaf_curl"], "PNG")

    # 5. Bacterial Leaf Spot (35-50 small angular water-soaked dark brown spots)
    img_bs, leaf_mask = _draw_leaf_base()
    draw_bs = ImageDraw.Draw(img_bs)
    np.random.seed(101)
    # Generate spots clustered mostly inside leaf blade
    for _ in range(46):
        bx = np.random.randint(145, 255)
        by = np.random.randint(90, 325)
        if leaf_mask[by, bx]:
            spot_w = np.random.randint(3, 8)
            spot_h = np.random.randint(3, 8)
            # Water-soaked yellowish halo
            draw_bs.ellipse([bx - spot_w - 2, by - spot_h - 2, bx + spot_w + 2, by + spot_h + 2], fill=(160, 145, 45))
            # Dark necrotic angular core
            draw_bs.rectangle([bx - spot_w // 2, by - spot_h // 2, bx + spot_w // 2, by + spot_h // 2], fill=(38, 22, 14))
    img_bs.save(manifest["bacterial_spot"], "PNG")

    # 6. Late Blight (Large dark water-soaked olive-brown/black spreading lesions)
    img_lb, leaf_mask = _draw_leaf_base()
    lb_arr = np.array(img_lb).astype(np.float32)
    lb_centers = [(150, 110), (245, 170), (145, 235), (230, 280), (185, 80)]
    for cx, cy in lb_centers:
        rx = np.random.uniform(32.0, 48.0)
        ry = np.random.uniform(28.0, 42.0)
        dist_sq = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
        blight_mask = np.clip(1.0 - dist_sq, 0.0, 1.0)
        # Irregular shape
        blight_mask = blight_mask ** 1.35
        intensity = (blight_mask * 0.94)[:, :, np.newaxis]
        # Dark necrotic olive-black (36, 32, 22)
        necrotic_color = np.array([38.0, 34.0, 24.0], dtype=np.float32)
        lb_arr = lb_arr * (1.0 - intensity) + necrotic_color * intensity

    lb_final = np.clip(lb_arr, 0, 255).astype(np.uint8)
    lb_final[~leaf_mask] = base_bg[~leaf_mask]
    Image.fromarray(lb_final).save(manifest["late_blight"], "PNG")

    logger.info("Successfully generated all 6 procedural sample leaf images.")
    return {k: str(v.resolve()) for k, v in manifest.items()}


# ============================================================================
# 4. COMPUTER VISION FEATURE EXTRACTION & TIER 2 ENGINE
# ============================================================================

def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB to HSV color-space converter in pure NumPy (Zero OpenCV dependency).
    Input: rgb array shaped (H, W, 3) in range [0, 255].
    Output: hsv array shaped (H, W, 3) where:
      - H in [0, 360] degrees
      - S in [0.0, 1.0]
      - V in [0.0, 1.0]
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    # Hue calculation
    h = np.zeros_like(c_max)

    # When delta > 0
    mask_r = (c_max == r) & (delta > 1e-6)
    mask_g = (c_max == g) & (delta > 1e-6)
    mask_b = (c_max == b) & (delta > 1e-6)

    h[mask_r] = (60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0))
    h[mask_g] = (60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0))
    h[mask_b] = (60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0))

    # Saturation
    s = np.zeros_like(c_max)
    nz_v = c_max > 1e-6
    s[nz_v] = delta[nz_v] / c_max[nz_v]

    # Value
    v = c_max

    return np.stack([h, s, v], axis=-1)


class PlantDiseaseEngine:
    """
    Production-grade Plant Leaf Disease Prediction & Diagnostic Engine.
    Implements a robust Dual-Tier architecture:
    - Tier 1: Optional HuggingFace MobileNetV2 Vision Transformer/CNN pipeline
    - Tier 2: Pure-Python CV Color-Space Feature Extraction + Random Forest Classifier (100% Offline)
    """

    CLASS_LABELS = [
        "healthy",
        "cercospora_beet",
        "early_blight",
        "powdery_mildew",
        "yellow_leaf_curl",
        "bacterial_spot",
        "late_blight",
    ]

    def __init__(self, enable_tier1: bool = True):
        self.tier1_available = False
        self.tier1_pipeline = None
        self.rf_classifier: Optional[RandomForestClassifier] = None

        logger.info("Initializing PhytoScan AI Diagnostic Engine...")
        self._init_tier1(enable_tier1)
        self._init_tier2_classifier()
        logger.info("PhytoScan AI Engine ready.")

    def _init_tier1(self, enable: bool):
        """Attempts to load HuggingFace MobileNetV2 Plant Disease model if installed & available."""
        if not enable:
            logger.info("Tier 1 HuggingFace pipeline disabled by configuration.")
            return

        try:
            from transformers import pipeline
            logger.info("Attempting Tier 1 HuggingFace vision pipeline initialization...")
            self.tier1_pipeline = pipeline(
                "image-classification",
                model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
            )
            self.tier1_available = True
            logger.info("Tier 1 HuggingFace MobileNetV2 pipeline loaded successfully.")
        except Exception as exc:
            self.tier1_available = False
            logger.info(f"Tier 1 HuggingFace not available ({exc}). Defaulting to 100% offline Tier 2 Engine.")

    def _init_tier2_classifier(self):
        """
        Trains a robust multi-class Random Forest Classifier on synthetic feature vectors
        representing the 7 botanical disease states.
        Feature vector:
        [0]: green_ratio
        [1]: brown_ratio
        [2]: yellow_ratio
        [3]: white_ratio
        [4]: lesion_cluster_variance
        [5]: dark_necrosis_ratio
        [6]: color_entropy
        """
        logger.info("Training Tier 2 Random Forest Classifier on feature-space distributions...")
        rng = np.random.RandomState(42)
        n_samples_per_class = 150

        X_list = []
        y_list = []

        # 0: Healthy (high green, near-zero brown/yellow/white)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.95, 1.00)
            b = rng.uniform(0.00, 0.01)
            y = rng.uniform(0.00, 0.01)
            w = rng.uniform(0.00, 0.01)
            var = rng.uniform(0.00, 0.01)
            dark = rng.uniform(0.00, 0.01)
            ent = rng.uniform(0.20, 0.28)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(0)

        # 1: Cercospora Leaf Spot (Sugar Beet: b in 0.04-0.11, y in 0.12-0.24, w in 0.01-0.05, var in 0.03-0.22, dark in 0.00-0.015, g in 0.65-0.79)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.65, 0.79)
            b = rng.uniform(0.04, 0.11)
            y = rng.uniform(0.12, 0.24)
            w = rng.uniform(0.01, 0.05)
            var = rng.uniform(0.03, 0.22)
            dark = rng.uniform(0.00, 0.015)
            ent = rng.uniform(0.26, 0.33)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(1)

        # 2: Early Blight (b in 0.09-0.18, y in 0.07-0.14, g in 0.68-0.82)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.68, 0.82)
            b = rng.uniform(0.09, 0.18)
            y = rng.uniform(0.07, 0.14)
            w = rng.uniform(0.00, 0.02)
            var = rng.uniform(0.01, 0.06)
            dark = rng.uniform(0.015, 0.04)
            ent = rng.uniform(0.26, 0.33)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(2)

        # 3: Powdery Mildew (w in 0.15-0.45, g in 0.55-0.80)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.55, 0.80)
            b = rng.uniform(0.00, 0.02)
            y = rng.uniform(0.00, 0.03)
            w = rng.uniform(0.15, 0.45)
            var = rng.uniform(0.01, 0.05)
            dark = rng.uniform(0.00, 0.02)
            ent = rng.uniform(0.24, 0.30)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(3)

        # 4: Yellow Leaf Curl (y in 0.25-0.55, g in 0.45-0.72, b < 0.03)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.40, 0.68)
            b = rng.uniform(0.00, 0.03)
            y = rng.uniform(0.28, 0.58)
            w = rng.uniform(0.00, 0.02)
            var = rng.uniform(0.02, 0.08)
            dark = rng.uniform(0.00, 0.02)
            ent = rng.uniform(0.28, 0.35)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(4)

        # 5: Bacterial Leaf Spot (b in 0.015-0.045, y in 0.08-0.18, dark in 0.015-0.04, g in 0.78-0.89)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.78, 0.89)
            b = rng.uniform(0.015, 0.045)
            y = rng.uniform(0.08, 0.18)
            w = rng.uniform(0.00, 0.02)
            var = rng.uniform(0.01, 0.05)
            dark = rng.uniform(0.015, 0.04)
            ent = rng.uniform(0.25, 0.31)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(5)

        # 6: Late Blight (b in 0.14-0.30, dark in 0.06-0.20, y < 0.03, g in 0.70-0.85)
        for _ in range(n_samples_per_class):
            g = rng.uniform(0.70, 0.85)
            b = rng.uniform(0.14, 0.30)
            y = rng.uniform(0.00, 0.03)
            w = rng.uniform(0.00, 0.02)
            var = rng.uniform(0.02, 0.07)
            dark = rng.uniform(0.06, 0.20)
            ent = rng.uniform(0.25, 0.32)
            X_list.append([g, b, y, w, var, dark, ent])
            y_list.append(6)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        self.rf_classifier = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
        self.rf_classifier.fit(X, y)
        logger.info(f"Tier 2 Random Forest trained successfully on {len(X)} feature vectors.")

    def _segment_leaf_blade(self, rgb_arr: np.ndarray) -> np.ndarray:
        """
        Segments the foreground leaf lamina from background using adaptive color and luminance thresholding.
        Returns a boolean 2D numpy mask where True indicates leaf blade pixels.
        """
        hsv = _rgb_to_hsv_numpy(rgb_arr)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r = rgb_arr[..., 0].astype(np.float32)
        g = rgb_arr[..., 1].astype(np.float32)
        b = rgb_arr[..., 2].astype(np.float32)

        # Sample corner background color
        h_dim, w_dim = rgb_arr.shape[:2]
        corners = np.concatenate([
            rgb_arr[0:15, 0:15].reshape(-1, 3),
            rgb_arr[0:15, w_dim - 15:w_dim].reshape(-1, 3),
            rgb_arr[h_dim - 15:h_dim, 0:15].reshape(-1, 3),
            rgb_arr[h_dim - 15:h_dim, w_dim - 15:w_dim].reshape(-1, 3),
        ], axis=0)
        bg_mean = np.mean(corners, axis=0)

        # Distance from background color
        diff_bg = np.linalg.norm(rgb_arr.astype(np.float32) - bg_mean, axis=-1)

        # Leaf criteria:
        # 1. Distinct from background
        # 2. Plant pigments (green hue band, yellow chlorosis band, brown necrosis, or powdery mildew inside leaf)
        is_greenish = (h >= 60.0) & (h <= 170.0) & (s >= 0.15) & (v >= 0.12)
        is_yellowish = (h >= 35.0) & (h < 60.0) & (s >= 0.22) & (v >= 0.30)
        is_brownish = (h >= 10.0) & (h < 45.0) & (s >= 0.20) & (v <= 0.60) & (r > b)
        is_dark_lesion = (v <= 0.35) & (r >= b) & (diff_bg > 25.0)
        is_mycelium = (s <= 0.32) & (v >= 0.58) & (diff_bg > 30.0)

        leaf_mask = (diff_bg > 18.0) & (is_greenish | is_yellowish | is_brownish | is_dark_lesion | is_mycelium)

        # Fallback safeguard: if leaf_mask is too small (e.g. extreme white or dark background)
        if np.sum(leaf_mask) < 2000:
            leaf_mask = diff_bg > 28.0

        return leaf_mask

    def _extract_cv_features(self, rgb_arr: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
        """
        Extracts multi-spectral color-space features and lesion masks from the segmented leaf.
        Returns:
        - feature_vector (7-dim numpy array)
        - masks_dict (healthy_green, necrotic_brown, chlorotic_yellow, fungal_white)
        - metrics_dict (percentages of total leaf area)
        """
        hsv = _rgb_to_hsv_numpy(rgb_arr)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r = rgb_arr[..., 0].astype(np.float32) / 255.0
        g = rgb_arr[..., 1].astype(np.float32) / 255.0
        b = rgb_arr[..., 2].astype(np.float32) / 255.0

        n_leaf = max(1, int(np.sum(leaf_mask)))

        # 1. Necrotic Brown / Black Mask (low brightness, target-rings, water-soaked rot)
        necrotic_brown = leaf_mask & (
            ((v < 0.32) & (r >= g * 0.85) & (r > b * 1.05)) |
            ((h >= 10.0) & (h <= 42.0) & (s >= 0.22) & (v <= 0.52)) |
            ((r > 0.08) & (r < 0.45) & (g < 0.35) & (b < 0.25) & (r > b * 1.15))
        )

        # 2. White / Gray Fungal Mycelium Mask (powdery mildew)
        fungal_white = leaf_mask & (~necrotic_brown) & (
            (s <= 0.28) & (v >= 0.65) & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 0.22)
        )

        # 3. Chlorotic Yellow Mask (yellow leaf curl, chlorotic halo)
        chlorotic_yellow = leaf_mask & (~necrotic_brown) & (~fungal_white) & (
            ((h >= 36.0) & (h <= 64.0) & (s >= 0.25) & (v >= 0.35)) |
            ((r + g > 2.05 * b) & (g > 0.40) & (r > 0.35) & (h < 68.0))
        )

        # 4. Healthy Green Mask
        healthy_green = leaf_mask & (~necrotic_brown) & (~fungal_white) & (~chlorotic_yellow) & (
            (h >= 62.0) & (h <= 165.0) & (g > r * 0.95) & (g > b * 1.05) & (s >= 0.16)
        )

        # Calculate exact pixel area percentages relative to total leaf blade
        n_green = int(np.sum(healthy_green))
        n_brown = int(np.sum(necrotic_brown))
        n_yellow = int(np.sum(chlorotic_yellow))
        n_white = int(np.sum(fungal_white))

        green_ratio = n_green / n_leaf
        brown_ratio = n_brown / n_leaf
        yellow_ratio = n_yellow / n_leaf
        white_ratio = n_white / n_leaf

        # Feature 5: Spatial Lesion Cluster Variance
        diseased_mask = necrotic_brown | chlorotic_yellow | fungal_white
        y_coords, x_coords = np.where(diseased_mask)
        if len(y_coords) > 20:
            var_y = np.var(y_coords) / (rgb_arr.shape[0] ** 2)
            var_x = np.var(x_coords) / (rgb_arr.shape[1] ** 2)
            lesion_cluster_var = float(np.clip(var_y + var_x, 0.0, 1.0))
        else:
            lesion_cluster_var = 0.0

        # Feature 6: Dark Necrosis Ratio (extreme necrotic tissue)
        dark_necrosis_mask = leaf_mask & (v < 0.22) & (r >= b)
        dark_necrosis_ratio = float(np.sum(dark_necrosis_mask) / n_leaf)

        # Feature 7: Color Entropy within leaf blade
        hue_leaf = h[leaf_mask]
        hist, _ = np.histogram(hue_leaf, bins=18, range=(0, 360), density=True)
        hist = hist[hist > 0]
        color_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.5

        feature_vector = np.array([
            green_ratio,
            brown_ratio,
            yellow_ratio,
            white_ratio,
            lesion_cluster_var,
            dark_necrosis_ratio,
            color_entropy,
        ], dtype=np.float32)

        masks_dict = {
            "leaf": leaf_mask,
            "green": healthy_green,
            "brown": necrotic_brown,
            "yellow": chlorotic_yellow,
            "white": fungal_white,
        }

        metrics_dict = {
            "Healthy Green %": round(green_ratio * 100.0, 1),
            "Brown Necrosis %": round(brown_ratio * 100.0, 1),
            "Yellow Chlorosis %": round(yellow_ratio * 100.0, 1),
            "White Fungal %": round(white_ratio * 100.0, 1),
            "Total Infected Area %": round(min(100.0, (brown_ratio + yellow_ratio + white_ratio) * 100.0), 1),
        }

        return feature_vector, masks_dict, metrics_dict

    def _generate_lesion_heatmap_overlay(
        self,
        original_rgb: np.ndarray,
        masks: Dict[str, np.ndarray],
        metrics: Dict[str, float]
    ) -> Image.Image:
        """
        Creates an enterprise-grade visual AI Lesion Segmentation Map / Diagnostic Heatmap.
        - Healthy tissue: natural subdued green
        - Brown Necrosis: bright crimson red (#EF4444) with contour glow
        - Chlorosis: bright amber/gold (#F59E0B)
        - Powdery Mildew: electric violet (#A855F7)
        - Background: darkened neutral slate
        Includes an agricultural diagnostic legend overlay at the bottom.
        """
        overlay = original_rgb.astype(np.float32).copy()
        leaf_mask = masks["leaf"]
        green_mask = masks["green"]
        brown_mask = masks["brown"]
        yellow_mask = masks["yellow"]
        white_mask = masks["white"]

        # Subdue non-leaf background to dark slate
        bg_dim = np.array([20.0, 24.0, 30.0], dtype=np.float32)
        overlay[~leaf_mask] = overlay[~leaf_mask] * 0.25 + bg_dim * 0.75

        # Healthy lamina: slightly enhance contrast
        overlay[green_mask] = overlay[green_mask] * 0.85 + np.array([22.0, 145.0, 52.0], dtype=np.float32) * 0.15

        # Necrotic Brown -> Vivid Crimson Red (239, 68, 68) with 75% blend
        red_tint = np.array([239.0, 68.0, 68.0], dtype=np.float32)
        overlay[brown_mask] = overlay[brown_mask] * 0.25 + red_tint * 0.75

        # Chlorotic Yellow -> Radiant Amber/Orange (245, 158, 11) with 75% blend
        amber_tint = np.array([245.0, 158.0, 11.0], dtype=np.float32)
        overlay[yellow_mask] = overlay[yellow_mask] * 0.25 + amber_tint * 0.75

        # Powdery White -> Electric Violet/Magenta (168, 85, 247) with 75% blend
        violet_tint = np.array([168.0, 85.0, 247.0], dtype=np.float32)
        overlay[white_mask] = overlay[white_mask] * 0.25 + violet_tint * 0.75

        overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

        # Draw visual inspection banner and legend at bottom of canvas
        draw = ImageDraw.Draw(overlay_img)
        w, h = overlay_img.size
        legend_h = 36
        # Semi-transparent dark banner bar
        banner_rect = [0, h - legend_h, w, h]
        banner_overlay = Image.new("RGBA", (w, legend_h), (15, 23, 42, 230))
        overlay_img.paste(Image.new("RGB", (w, legend_h), (15, 23, 42)), (0, h - legend_h))
        draw = ImageDraw.Draw(overlay_img)

        # Legend items
        legend_items = [
            ("Healthy", (34, 197, 94)),
            ("Necrosis", (239, 68, 68)),
            ("Chlorosis", (245, 158, 11)),
            ("Mycelium", (168, 85, 247)),
        ]

        x_cursor = 14
        y_pos = h - legend_h + 10
        for label, color in legend_items:
            # Draw color swatch circle
            draw.ellipse([x_cursor, y_pos + 1, x_cursor + 12, y_pos + 13], fill=color, outline=(255, 255, 255))
            draw.text((x_cursor + 18, y_pos), label, fill=(241, 245, 249))
            x_cursor += 92

        return overlay_img

    def analyze_image(self, image_source: str | Image.Image) -> PredictionResult:
        """
        Executes end-to-end diagnostic pipeline:
        1. Loads & normalizes leaf image to 400x400 RGB.
        2. Segments leaf blade from background.
        3. Extracts color-space spectral features (HSV/RGB).
        4. Synthesizes AI Lesion Segmentation Map Heatmap overlay.
        5. Computes classification probabilities, severity index, and agronomic treatment plan.
        """
        start_time = time.perf_counter()

        # Load image
        if isinstance(image_source, (str, Path)):
            pil_img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image_source type: {type(image_source)}")

        # Safe Pillow 10+ resize
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        normalized_img = pil_img.resize((400, 400), resample_filter)
        rgb_arr = np.array(normalized_img)

        # 1. Segment Leaf Blade
        leaf_mask = self._segment_leaf_blade(rgb_arr)

        # 2. Extract CV Features & Metrics
        features, masks, metrics = self._extract_cv_features(rgb_arr, leaf_mask)

        # 3. Generate Lesion Segmentation Heatmap
        heatmap_overlay = self._generate_lesion_heatmap_overlay(rgb_arr, masks, metrics)

        # 4. Multi-Tier Classification
        engine_used = "Tier 2 (Pure CV + Random Forest)"
        top_3: List[Tuple[str, float]] = []

        # Attempt Tier 1 if available
        tier1_succeeded = False
        if self.tier1_available and self.tier1_pipeline is not None:
            try:
                hf_preds = self.tier1_pipeline(normalized_img)
                # Map HF labels if recognized
                top_item = hf_preds[0]
                top_label = top_item["label"].lower()
                top_score = float(top_item["score"]) * 100.0

                matched_id = None
                for candidate in self.CLASS_LABELS:
                    if candidate in top_label:
                        matched_id = candidate
                        break

                if matched_id:
                    tier1_succeeded = True
                    engine_used = "Tier 1 (HuggingFace MobileNetV2)"
                    # Create top 3 from HF
                    for item in hf_preds[:3]:
                        top_3.append((item["label"], round(float(item["score"]) * 100.0, 1)))
                    predicted_id = matched_id
                    confidence_pct = round(top_score, 1)
            except Exception as e:
                logger.warning(f"Tier 1 inference failed: {e}. Falling back to Tier 2.")

        # Tier 2 Fallback / Primary Classification
        if not tier1_succeeded:
            # Predict probabilities via Random Forest
            rf_probs = self.rf_classifier.predict_proba([features])[0]

            # Domain Calibration based on spectral physics:
            # - If leaf has negligible lesions and >85% green -> healthy dominates
            green_r = features[0]
            brown_r = features[1]
            yellow_r = features[2]
            white_r = features[3]
            dark_r = features[5]

            calibrated_probs = rf_probs.copy()
            if green_r >= 0.88 and brown_r < 0.03 and yellow_r < 0.05 and white_r < 0.03:
                calibrated_probs[0] += 0.80  # Healthy boost
            if white_r >= 0.12:
                calibrated_probs[2] += 0.65  # Powdery Mildew boost
            if yellow_r >= 0.25 and brown_r < 0.10:
                calibrated_probs[3] += 0.60  # Yellow Leaf Curl boost
            if brown_r >= 0.10 and features[4] < 0.25 and dark_r >= 0.05:
                calibrated_probs[4] += 0.45  # Bacterial Leaf Spot
            if dark_r >= 0.15 and brown_r >= 0.18:
                calibrated_probs[5] += 0.50  # Late Blight
            if brown_r >= 0.10 and yellow_r >= 0.06:
                calibrated_probs[1] += 0.40  # Early Blight target halo

            # Re-normalize probabilities
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

            # Sort top 3
            ranked_indices = np.argsort(calibrated_probs)[::-1]
            top_3 = []
            for idx in ranked_indices[:3]:
                d_id = self.CLASS_LABELS[idx]
                disp_name = DISEASE_DATABASE[d_id].name
                prob_pct = round(float(calibrated_probs[idx]) * 100.0, 1)
                top_3.append((disp_name, prob_pct))

            predicted_id = self.CLASS_LABELS[ranked_indices[0]]
            confidence_pct = round(float(calibrated_probs[ranked_indices[0]]) * 100.0, 1)

        # 5. Severity Assessment
        infected_pct = metrics["Total Infected Area %"]
        if infected_pct < 1.0:
            severity_level = "None (Healthy)"
        elif infected_pct <= 10.0:
            severity_level = "Mild (1–10%)"
        elif infected_pct <= 30.0:
            severity_level = "Moderate (11–30%)"
        else:
            severity_level = "Severe (>30%)"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
        profile = DISEASE_DATABASE.get(predicted_id, DISEASE_DATABASE["healthy"])

        return PredictionResult(
            predicted_disease=profile,
            confidence_pct=confidence_pct,
            top_3_predictions=top_3,
            infected_area_pct=infected_pct,
            severity_level=severity_level,
            feature_metrics=metrics,
            original_image=normalized_img,
            segmented_overlay_image=heatmap_overlay,
            engine_used=engine_used,
            processing_time_ms=elapsed_ms,
        )


# Global singleton instance for high-speed reuse
_global_engine: Optional[PlantDiseaseEngine] = None

def get_engine() -> PlantDiseaseEngine:
    """Returns the shared global instance of PlantDiseaseEngine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = PlantDiseaseEngine(enable_tier1=True)
    return _global_engine
