"""
PhytoScan AI — Desktop Diagnostic GUI Application
=================================================
Enterprise-grade Tkinter Agricultural Diagnostic Dashboard for Plant Leaf Disease
Prediction & Pathology Analysis.

Features:
- Dual high-resolution preview canvases (Original Specimen vs. AI Lesion Heatmap)
- Real-time tissue breakdown & Top-3 differential diagnosis table
- Agronomic pathology knowledge base with organic/chemical treatment plans
- Non-blocking background worker thread with safe UI marshaling
- Report export capability (.txt)
- Auto-generates and loads realistic sample leaves on startup
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

# Ensure engine can be imported if app.py is executed directly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from engine import (
    PlantDiseaseEngine,
    PredictionResult,
    DiseaseProfile,
    generate_sample_leaf_images,
    get_engine,
)


# Safe Pillow 10+ resampling constant
LANCZOS_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


class PlantDiseaseApp:
    """
    Main Tkinter Desktop Application for PhytoScan AI Plant Leaf Disease Prediction & Diagnostic System.
    """

    SAMPLE_LABELS = {
        "01 - Healthy Leaf": "healthy",
        "02 - Cercospora Leaf Spot (Sugar Beet)": "cercospora_beet",
        "03 - Early Blight (Tomato/Potato)": "early_blight",
        "04 - Powdery Mildew (Squash)": "powdery_mildew",
        "05 - Yellow Leaf Curl Virus (Tomato)": "yellow_leaf_curl",
        "06 - Bacterial Leaf Spot (Pepper)": "bacterial_spot",
        "07 - Late Blight (Potato)": "late_blight",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PhytoScan AI — Plant Leaf Disease Prediction & Diagnostic System")
        self.root.geometry("1180x780")
        self.root.minsize(980, 660)

        # Background palette
        self.bg_main = "#F8FAFC"      # Slate-50
        self.card_bg = "#FFFFFF"      # Pure White
        self.border_col = "#E2E8F0"   # Slate-200
        self.text_primary = "#0F172A" # Slate-900
        self.text_muted = "#64748B"   # Slate-500
        self.accent_green = "#16A34A" # Emerald Green
        self.accent_blue = "#2563EB"  # Royal Blue

        self.root.configure(bg=self.bg_main)

        # Persistent PhotoImage references to prevent Python Garbage Collection
        self._current_orig_photo: Optional[ImageTk.PhotoImage] = None
        self._current_overlay_photo: Optional[ImageTk.PhotoImage] = None
        self._current_raw_pil: Optional[Image.Image] = None
        self._current_image_path: Optional[str] = None
        self._last_result: Optional[PredictionResult] = None

        # Diagnostic Engine & Sample Leaf initialization
        self.engine: PlantDiseaseEngine = get_engine()
        samples_dir = current_dir / "sample_leaves"
        self.sample_paths: Dict[str, str] = generate_sample_leaf_images(str(samples_dir))

        # Setup GUI Components
        self._configure_styles()
        self._build_header()
        self._build_toolbar()
        self._build_workspace()
        self._build_statusbar()

        # Load first sample automatically on startup for instant live demo
        self.root.after(150, self._load_initial_sample)

    def _configure_styles(self):
        """Configures ttk styles for modern clean appearance."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.bg_main, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("Toolbar.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("Header.TFrame", background=self.card_bg, relief="solid", borderwidth=1)

        # Buttons
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 9, "bold"),
            background=self.accent_blue,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6,
        )
        style.map("Primary.TButton", background=[("active", "#1D4ED8"), ("disabled", "#94A3B8")])

        style.configure(
            "Success.TButton",
            font=("Segoe UI", 9, "bold"),
            background=self.accent_green,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6,
        )
        style.map("Success.TButton", background=[("active", "#15803D"), ("disabled", "#94A3B8")])

        style.configure(
            "Outline.TButton",
            font=("Segoe UI", 9),
            background=self.card_bg,
            foreground=self.text_primary,
            borderwidth=1,
            padding=6,
        )

        # Treeview styling
        style.configure(
            "Treeview",
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground=self.text_primary,
            rowheight=24,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#F1F5F9",
            foreground=self.text_primary,
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#E2E8F0")], foreground=[("selected", self.text_primary)])

        # Notebook tabs
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 9, "bold"),
            padding=[12, 6],
            background="#E2E8F0",
            foreground=self.text_muted,
        )
        style.map("TNotebook.Tab", background=[("selected", self.card_bg)], foreground=[("selected", self.text_primary)])

    def _build_header(self):
        """Builds top enterprise-grade header bar."""
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=64, padx=18, pady=10, relief="solid", bd=1)
        header_frame.pack(side="top", fill="x", padx=12, pady=(10, 6))

        # Title container
        title_box = tk.Frame(header_frame, bg=self.card_bg)
        title_box.pack(side="left", fill="y")

        lbl_title = tk.Label(
            title_box,
            text="PhytoScan AI — Plant Leaf Disease Prediction & Diagnostic System",
            font=("Segoe UI", 13, "bold"),
            fg=self.text_primary,
            bg=self.card_bg,
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text="Computer Vision & Machine Learning Internal Assessment Project | Dual-Tier Hybrid Architecture",
            font=("Segoe UI", 9),
            fg=self.text_muted,
            bg=self.card_bg,
        )
        lbl_sub.pack(anchor="w")

        # Right-aligned Engine Ready Badge
        badge_box = tk.Frame(header_frame, bg="#DCFCE7", padx=12, pady=5, relief="solid", bd=1)
        badge_box.pack(side="right", anchor="center")

        engine_tier_text = "[Engine Ready] Hybrid CV + Random Forest Classifier (7 Classes)"
        lbl_badge = tk.Label(
            badge_box,
            text=engine_tier_text,
            font=("Segoe UI", 8, "bold"),
            fg="#166534",
            bg="#DCFCE7",
        )
        lbl_badge.pack()

    def _build_toolbar(self):
        """Builds action toolbar containing demo presets, upload, diagnose, and export buttons."""
        toolbar = tk.Frame(self.root, bg=self.card_bg, padx=14, pady=8, relief="solid", bd=1)
        toolbar.pack(side="top", fill="x", padx=12, pady=(0, 6))

        # Demo Sample Dropdown
        lbl_demo = tk.Label(toolbar, text="Load Demo Leaf Sample:", font=("Segoe UI", 9, "bold"), bg=self.card_bg, fg=self.text_primary)
        lbl_demo.pack(side="left", padx=(0, 6))

        self.combo_sample = ttk.Combobox(
            toolbar,
            values=list(self.SAMPLE_LABELS.keys()),
            state="readonly",
            width=32,
            font=("Segoe UI", 9),
        )
        self.combo_sample.current(0)
        self.combo_sample.pack(side="left", padx=(0, 10))
        self.combo_sample.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # Primary Action: Run Diagnosis
        self.btn_diagnose = ttk.Button(
            toolbar,
            text="Run Disease Diagnosis",
            style="Primary.TButton",
            command=self.run_diagnosis,
        )
        self.btn_diagnose.pack(side="left", padx=4)

        # Upload Button
        self.btn_upload = ttk.Button(
            toolbar,
            text="Upload Leaf Image (.jpg/.png)",
            style="Success.TButton",
            command=self._on_upload_image,
        )
        self.btn_upload.pack(side="left", padx=4)

        # Export Report Button
        self.btn_export = ttk.Button(
            toolbar,
            text="Export Diagnostic Report (.txt)",
            style="Outline.TButton",
            command=self._on_export_report,
        )
        self.btn_export.pack(side="left", padx=4)

        # Reset / Clear Button
        self.btn_reset = ttk.Button(
            toolbar,
            text="Reset / Clear",
            style="Outline.TButton",
            command=self._on_reset,
        )
        self.btn_reset.pack(side="left", padx=4)

        # Progress Spinner / Label
        self.lbl_progress = tk.Label(
            toolbar,
            text="",
            font=("Segoe UI", 9, "italic"),
            bg=self.card_bg,
            fg=self.accent_blue,
        )
        self.lbl_progress.pack(side="right", padx=10)

    def _build_workspace(self):
        """Constructs 2-column main workspace layout."""
        workspace = tk.Frame(self.root, bg=self.bg_main)
        workspace.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 6))

        # ==================== LEFT COLUMN: VISUAL INSPECTION STUDIO ====================
        left_col = tk.Frame(workspace, bg=self.bg_main, width=540)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Dual Image Previews Card
        preview_card = tk.Frame(left_col, bg=self.card_bg, padx=10, pady=10, relief="solid", bd=1)
        preview_card.pack(side="top", fill="x", pady=(0, 8))

        lbl_pv_header = tk.Label(
            preview_card,
            text="Dual-Spectrum Visual Inspection Studio",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            fg=self.text_primary,
        )
        lbl_pv_header.pack(anchor="w", pady=(0, 6))

        canvases_box = tk.Frame(preview_card, bg=self.card_bg)
        canvases_box.pack(fill="x", expand=True)

        # Specimen Canvas (Card A)
        box_orig = tk.Frame(canvases_box, bg="#F1F5F9", relief="solid", bd=1)
        box_orig.pack(side="left", fill="both", expand=True, padx=(0, 5))

        lbl_a_title = tk.Label(box_orig, text="Original Leaf Specimen", font=("Segoe UI", 8, "bold"), bg="#E2E8F0", fg=self.text_primary)
        lbl_a_title.pack(side="top", fill="x")

        self.canvas_orig = tk.Canvas(box_orig, width=250, height=250, bg="#1E293B", highlightthickness=0)
        self.canvas_orig.pack(side="top", fill="both", expand=True, padx=4, pady=4)

        # Heatmap Overlay Canvas (Card B)
        box_heat = tk.Frame(canvases_box, bg="#F1F5F9", relief="solid", bd=1)
        box_heat.pack(side="left", fill="both", expand=True, padx=(5, 0))

        lbl_b_title = tk.Label(box_heat, text="AI Lesion Segmentation Map", font=("Segoe UI", 8, "bold"), bg="#E2E8F0", fg=self.text_primary)
        lbl_b_title.pack(side="top", fill="x")

        self.canvas_heat = tk.Canvas(box_heat, width=250, height=250, bg="#1E293B", highlightthickness=0)
        self.canvas_heat.pack(side="top", fill="both", expand=True, padx=4, pady=4)

        # Differential Diagnosis & Tissue Breakdown Card
        table_card = tk.Frame(left_col, bg=self.card_bg, padx=10, pady=10, relief="solid", bd=1)
        table_card.pack(side="top", fill="both", expand=True)

        lbl_tbl_title = tk.Label(
            table_card,
            text="Differential Diagnosis & Pixel Tissue Breakdown",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            fg=self.text_primary,
        )
        lbl_tbl_title.pack(anchor="w", pady=(0, 4))

        # Top 3 Diagnosis Table
        lbl_top3 = tk.Label(table_card, text="Top-3 Differential Diagnosis Probabilities:", font=("Segoe UI", 8, "bold"), bg=self.card_bg, fg=self.text_muted)
        lbl_top3.pack(anchor="w")

        cols_top3 = ("rank", "disease", "confidence")
        self.tree_top3 = ttk.Treeview(table_card, columns=cols_top3, show="headings", height=3)
        self.tree_top3.heading("rank", text="#")
        self.tree_top3.heading("disease", text="Suspected Disease Pathology")
        self.tree_top3.heading("confidence", text="Confidence %")
        self.tree_top3.column("rank", width=35, anchor="center")
        self.tree_top3.column("disease", width=340, anchor="w")
        self.tree_top3.column("confidence", width=95, anchor="center")
        self.tree_top3.pack(fill="x", pady=(2, 8))

        # Spectral Tissue Metrics Table
        lbl_metrics = tk.Label(table_card, text="Spectral Tissue Color-Space Breakdown:", font=("Segoe UI", 8, "bold"), bg=self.card_bg, fg=self.text_muted)
        lbl_metrics.pack(anchor="w")

        cols_metrics = ("category", "spectral_band", "percentage")
        self.tree_metrics = ttk.Treeview(table_card, columns=cols_metrics, show="headings", height=5)
        self.tree_metrics.heading("category", text="Tissue Category")
        self.tree_metrics.heading("spectral_band", text="Biophysical / Spectral Band")
        self.tree_metrics.heading("percentage", text="Leaf Area %")
        self.tree_metrics.column("category", width=160, anchor="w")
        self.tree_metrics.column("spectral_band", width=200, anchor="w")
        self.tree_metrics.column("percentage", width=95, anchor="center")
        self.tree_metrics.pack(fill="both", expand=True, pady=(2, 0))

        # ==================== RIGHT COLUMN: AGRONOMIC PATHOLOGY REPORT ====================
        right_col = tk.Frame(workspace, bg=self.bg_main)
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # Diagnostic Summary Banner Card
        self.banner_card = tk.Frame(right_col, bg=self.card_bg, padx=14, pady=12, relief="solid", bd=1)
        self.banner_card.pack(side="top", fill="x", pady=(0, 8))

        # Disease Name Header
        self.lbl_diag_disease = tk.Label(
            self.banner_card,
            text="Ready to Diagnose Leaf Specimen",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_bg,
            fg=self.text_primary,
            wraplength=480,
            justify="left",
        )
        self.lbl_diag_disease.pack(anchor="w")

        # Sub-banner with badges
        sub_badges_frame = tk.Frame(self.banner_card, bg=self.card_bg)
        sub_badges_frame.pack(anchor="w", pady=(4, 0), fill="x")

        self.lbl_badge_pathogen = tk.Label(
            sub_badges_frame,
            text="Pathogen: —",
            font=("Segoe UI", 8, "bold"),
            bg="#E2E8F0",
            fg="#334155",
            padx=8,
            pady=2,
            relief="solid",
            bd=1,
        )
        self.lbl_badge_pathogen.pack(side="left", padx=(0, 6))

        self.lbl_badge_conf = tk.Label(
            sub_badges_frame,
            text="Confidence: —",
            font=("Segoe UI", 8, "bold"),
            bg="#DBEAFE",
            fg="#1E40AF",
            padx=8,
            pady=2,
            relief="solid",
            bd=1,
        )
        self.lbl_badge_conf.pack(side="left", padx=(0, 6))

        self.lbl_badge_severity = tk.Label(
            sub_badges_frame,
            text="Infected Area: —",
            font=("Segoe UI", 8, "bold"),
            bg="#DCFCE7",
            fg="#166534",
            padx=8,
            pady=2,
            relief="solid",
            bd=1,
        )
        self.lbl_badge_severity.pack(side="left", padx=(0, 6))

        # Report Tabs Notebook Card
        report_card = tk.Frame(right_col, bg=self.card_bg, padx=10, pady=10, relief="solid", bd=1)
        report_card.pack(side="top", fill="both", expand=True)

        self.notebook = ttk.Notebook(report_card)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Symptoms
        self.txt_symptoms = ScrolledText(self.notebook, wrap="word", font=("Segoe UI", 9), relief="flat", padx=10, pady=10)
        self.notebook.add(self.txt_symptoms, text="Symptoms & Biomarkers")

        # Tab 2: Organic Treatment
        self.txt_organic = ScrolledText(self.notebook, wrap="word", font=("Segoe UI", 9), relief="flat", padx=10, pady=10)
        self.notebook.add(self.txt_organic, text="Organic Treatment")

        # Tab 3: Chemical Control
        self.txt_chemical = ScrolledText(self.notebook, wrap="word", font=("Segoe UI", 9), relief="flat", padx=10, pady=10)
        self.notebook.add(self.txt_chemical, text="Chemical Control")

        # Tab 4: Prevention & Management
        self.txt_prevention = ScrolledText(self.notebook, wrap="word", font=("Segoe UI", 9), relief="flat", padx=10, pady=10)
        self.notebook.add(self.txt_prevention, text="Prevention Protocols")

        # Tab 5: Full Report Log
        self.txt_full_log = ScrolledText(self.notebook, wrap="word", font=("Consolas", 8), relief="flat", padx=10, pady=10)
        self.notebook.add(self.txt_full_log, text="Pathology Report (.txt)")

        # Style tags inside scrolled texts
        for txt_widget in (self.txt_symptoms, self.txt_organic, self.txt_chemical, self.txt_prevention, self.txt_full_log):
            txt_widget.tag_configure("heading", font=("Segoe UI", 10, "bold"), foreground=self.text_primary, spacing3=6)
            txt_widget.tag_configure("bullet", font=("Segoe UI", 9), foreground="#334155", lmargin1=10, lmargin2=25, spacing1=3)
            txt_widget.tag_configure("highlight", font=("Segoe UI", 9, "bold"), foreground=self.accent_green)

    def _build_statusbar(self):
        """Builds bottom status bar showing image metadata and engine inference metrics."""
        self.statusbar = tk.Frame(self.root, bg="#0F172A", height=24, padx=10, relief="flat")
        self.statusbar.pack(side="bottom", fill="x")

        self.lbl_status_left = tk.Label(
            self.statusbar,
            text="Ready | No image loaded",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#0F172A",
        )
        self.lbl_status_left.pack(side="left", padx=10, pady=2)

        self.lbl_status_right = tk.Label(
            self.statusbar,
            text="Inference: — ms | Engine: Hybrid CV + Random Forest",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#0F172A",
        )
        self.lbl_status_right.pack(side="right", padx=10, pady=2)

    # =========================================================================
    # LOGIC & EVENT HANDLERS
    # =========================================================================

    def _load_initial_sample(self):
        """Loads and analyzes the first sample leaf on application startup."""
        self.combo_sample.current(0)
        self._on_sample_selected(None)

    def _on_sample_selected(self, _event):
        """Triggered when a demo sample is chosen in the dropdown."""
        selection = self.combo_sample.get()
        sample_key = self.SAMPLE_LABELS.get(selection, "healthy")
        image_path = self.sample_paths.get(sample_key)

        if image_path and os.path.exists(image_path):
            self._load_image_file(image_path)
            self.run_diagnosis()

    def _on_upload_image(self):
        """Opens file dialog for user to upload external leaf image."""
        file_path = filedialog.askopenfilename(
            title="Select Plant Leaf Image Specimen",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self._load_image_file(file_path)
            self.run_diagnosis()

    def _load_image_file(self, file_path: str):
        """Loads and displays original image onto specimen canvas."""
        try:
            pil_img = Image.open(file_path).convert("RGB")
            self._current_raw_pil = pil_img
            self._current_image_path = file_path

            # Render on original canvas
            self._display_on_canvas(self.canvas_orig, pil_img, is_orig=True)
            self.lbl_status_left.config(text=f"Loaded: {os.path.basename(file_path)} | Size: {pil_img.width}x{pil_img.height}")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image file:\n{e}")

    def _display_on_canvas(self, canvas: tk.Canvas, pil_img: Image.Image, is_orig: bool):
        """Scales image proportionally to fit canvas (250x250) and updates persistent PhotoImage."""
        c_w = 250
        c_h = 250
        w, h = pil_img.size

        ratio = min(c_w / w, c_h / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))

        resized = pil_img.resize((new_w, new_h), LANCZOS_RESAMPLE)
        photo = ImageTk.PhotoImage(resized)

        # Store persistent reference on instance
        if is_orig:
            self._current_orig_photo = photo
        else:
            self._current_overlay_photo = photo

        canvas.delete("all")
        # Center in canvas
        offset_x = (c_w - new_w) // 2
        offset_y = (c_h - new_h) // 2
        canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)

    def run_diagnosis(self):
        """Launches non-blocking background analysis thread."""
        if self._current_raw_pil is None:
            messagebox.showwarning("No Specimen", "Please upload or select a leaf specimen before running diagnosis.")
            return

        self.btn_diagnose.config(state="disabled")
        self.lbl_progress.config(text="Extracting spectral features & running inference...")

        # Worker thread
        worker = threading.Thread(target=self._diagnostic_worker, daemon=True)
        worker.start()

    def _diagnostic_worker(self):
        """Executes core CV and ML engine in background worker thread."""
        try:
            result = self.engine.analyze_image(self._current_raw_pil)
            # Schedule UI update on main thread
            self.root.after(0, self._on_diagnosis_complete, result)
        except Exception as exc:
            self.root.after(0, self._on_diagnosis_error, str(exc))

    def _on_diagnosis_complete(self, result: PredictionResult):
        """Updates all GUI elements with diagnostic findings."""
        self._last_result = result
        self.btn_diagnose.config(state="normal")
        self.lbl_progress.config(text="")

        # 1. Update AI Lesion Segmentation Map Canvas
        self._display_on_canvas(self.canvas_heat, result.segmented_overlay_image, is_orig=False)

        # 2. Update Diagnostic Banner
        profile = result.predicted_disease
        self.lbl_diag_disease.config(text=profile.name)

        # Pathogen Badge
        pathogen_colors = {
            "Fungal": ("#FEF3C7", "#92400E"),
            "Bacterial": ("#FEE2E2", "#991B1B"),
            "Viral": ("#EDE9FE", "#5B21B6"),
            "None (Healthy)": ("#DCFCE7", "#166534"),
        }
        p_bg, p_fg = pathogen_colors.get(profile.pathogen_type, ("#E2E8F0", "#334155"))
        self.lbl_badge_pathogen.config(text=f"Pathogen: {profile.pathogen_type}", bg=p_bg, fg=p_fg)

        # Confidence Badge
        self.lbl_badge_conf.config(text=f"Confidence: {result.confidence_pct:.1f}%")

        # Severity Badge
        sev_colors = {
            "None (Healthy)": ("#DCFCE7", "#166534"),
            "Mild (1–10%)": ("#FEF9C3", "#854D0E"),
            "Moderate (11–30%)": ("#FFEDD5", "#9A3412"),
            "Severe (>30%)": ("#FEE2E2", "#991B1B"),
        }
        s_bg, s_fg = sev_colors.get(result.severity_level, ("#E2E8F0", "#334155"))
        self.lbl_badge_severity.config(
            text=f"Infected: {result.infected_area_pct:.1f}% ({result.severity_level})",
            bg=s_bg,
            fg=s_fg,
        )

        # 3. Update Top-3 Treeview
        for item in self.tree_top3.get_children():
            self.tree_top3.delete(item)
        for rank, (name, conf) in enumerate(result.top_3_predictions, 1):
            self.tree_top3.insert("", "end", values=(f"#{rank}", name, f"{conf:.1f}%"))

        # 4. Update Spectral Metrics Treeview
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        spectral_descriptions = {
            "Healthy Green %": "Healthy Chlorophyll-A/B (Hue 65–165°)",
            "Brown Necrosis %": "Necrotic Lesion Tissue (Low-V / Dark Brown)",
            "Yellow Chlorosis %": "Chlorosis / Viral Yellowing (Hue 36–64°)",
            "White Fungal %": "Mycelium Sporulation (High-V / Low-S)",
            "Total Infected Area %": "Cumulative Foliar Pathological Area",
        }
        for category, pct in result.feature_metrics.items():
            desc = spectral_descriptions.get(category, "Spectral Feature")
            self.tree_metrics.insert("", "end", values=(category, desc, f"{pct:.1f}%"))

        # 5. Populate Structured Agronomic Report Tabs
        self._populate_report_tabs(result)

        # 6. Update Status Bar
        fname = os.path.basename(self._current_image_path) if self._current_image_path else "Specimen"
        self.lbl_status_left.config(
            text=f"Analyzed: {fname} | Severity: {result.severity_level} | Infected Area: {result.infected_area_pct:.1f}%"
        )
        self.lbl_status_right.config(
            text=f"Inference: {result.processing_time_ms:.1f} ms | Engine: {result.engine_used}"
        )

    def _on_diagnosis_error(self, err_msg: str):
        """Handles diagnosis failure gracefully."""
        self.btn_diagnose.config(state="normal")
        self.lbl_progress.config(text="")
        messagebox.showerror("Diagnosis Error", f"An error occurred during image processing:\n{err_msg}")

    def _populate_report_tabs(self, result: PredictionResult):
        """Populates ScrolledText tabs with detailed treatment and pathology data."""
        profile = result.predicted_disease

        # Tab 1: Symptoms
        self.txt_symptoms.delete("1.0", "end")
        self.txt_symptoms.insert("end", "OBSERVED SYMPTOMS & PATHOLOGICAL BIOMARKERS\n", "heading")
        self.txt_symptoms.insert("end", f"Pathology: {profile.name}\n\n", "highlight")
        self.txt_symptoms.insert("end", f"{profile.symptoms}\n\n")
        self.txt_symptoms.insert("end", f"Assessed Severity Level: {result.severity_level}\n", "heading")
        self.txt_symptoms.insert("end", f"Foliar infection coverage: {result.infected_area_pct:.1f}% of leaf blade lamina.\n")

        # Tab 2: Organic Treatment
        self.txt_organic.delete("1.0", "end")
        self.txt_organic.insert("end", "ORGANIC & ECO-FRIENDLY BIO-MANAGEMENT PROTOCOL\n", "heading")
        self.txt_organic.insert("end", "Non-toxic and biological interventions for sustainable agricultural protection:\n\n")
        for tip in profile.organic_treatment:
            self.txt_organic.insert("end", f"•  {tip}\n", "bullet")

        # Tab 3: Chemical Control
        self.txt_chemical.delete("1.0", "end")
        self.txt_chemical.insert("end", "CHEMICAL & SYNTHETIC CONTROL PROTOCOL\n", "heading")
        self.txt_chemical.insert("end", "Targeted agricultural fungicides and bactericides for commercial yield protection:\n\n")
        for tip in profile.chemical_treatment:
            self.txt_chemical.insert("end", f"•  {tip}\n", "bullet")

        # Tab 4: Prevention Protocols
        self.txt_prevention.delete("1.0", "end")
        self.txt_prevention.insert("end", "PREVENTIVE CROP SANITATION & CULTURAL PRACTICES\n", "heading")
        self.txt_prevention.insert("end", "Long-term agronomic practices to prevent pathogen overwintering and spore spread:\n\n")
        for tip in profile.prevention_tips:
            self.txt_prevention.insert("end", f"•  {tip}\n", "bullet")

        # Tab 5: Formatted Full Pathology Report Log (.txt)
        report_text = self._generate_report_text(result)
        self.txt_full_log.delete("1.0", "end")
        self.txt_full_log.insert("end", report_text)

    def _generate_report_text(self, result: PredictionResult) -> str:
        """Generates a complete formatted plain text agricultural pathology report."""
        profile = result.predicted_disease
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname = os.path.basename(self._current_image_path) if self._current_image_path else "Unknown"

        sep = "=" * 76
        sub_sep = "-" * 76

        lines = [
            sep,
            "           PHYTOSCAN AI - AGRICULTURAL PLANT PATHOLOGY REPORT",
            sep,
            f"Report Timestamp       : {ts}",
            f"Specimen File Name     : {fname}",
            f"Diagnostic Engine Used : {result.engine_used}",
            f"Processing Latency     : {result.processing_time_ms:.1f} ms",
            sub_sep,
            "1. PRIMARY DIAGNOSIS SUMMARY",
            sub_sep,
            f"Detected Pathology     : {profile.name}",
            f"Pathogen Classification: {profile.pathogen_type}",
            f"Diagnostic Confidence  : {result.confidence_pct:.1f}%",
            f"Assessed Severity Level: {result.severity_level}",
            f"Total Infected Area    : {result.infected_area_pct:.1f}% of total leaf lamina",
            "",
            sub_sep,
            "2. TOP-3 DIFFERENTIAL DIAGNOSIS RANKINGS",
            sub_sep,
        ]
        for rank, (name, conf) in enumerate(result.top_3_predictions, 1):
            lines.append(f"  #{rank}  {name:<54} : {conf:.1f}%")

        lines.extend([
            "",
            sub_sep,
            "3. COMPUTER VISION COLOR-SPACE TISSUE BREAKDOWN",
            sub_sep,
        ])
        for cat, pct in result.feature_metrics.items():
            lines.append(f"  • {cat:<26}: {pct:>5.1f}%")

        lines.extend([
            "",
            sub_sep,
            "4. PATHOLOGICAL SYMPTOMS & CLINICAL BIOMARKERS",
            sub_sep,
            f"{profile.symptoms}",
            "",
            sub_sep,
            "5. ORGANIC & BIOLOGICAL TREATMENT RECOMMENDATIONS",
            sub_sep,
        ])
        for tip in profile.organic_treatment:
            lines.append(f"  [+] {tip}")

        lines.extend([
            "",
            sub_sep,
            "6. CHEMICAL / FUNGICIDAL CONTROL MEASURES",
            sub_sep,
        ])
        for tip in profile.chemical_treatment:
            lines.append(f"  [*] {tip}")

        lines.extend([
            "",
            sub_sep,
            "7. LONG-TERM PREVENTIVE & CULTURAL MEASURES",
            sub_sep,
        ])
        for tip in profile.prevention_tips:
            lines.append(f"  [>] {tip}")

        lines.extend([
            "",
            sep,
            " End of PhytoScan AI Diagnostic Report | University Internal Assessment",
            sep,
        ])
        return "\n".join(lines)

    def _on_export_report(self):
        """Exports the diagnostic pathology report to a formatted text file."""
        if not self._last_result:
            messagebox.showwarning("No Report Available", "Please run a diagnosis first before exporting a report.")
            return

        default_name = f"PhytoScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = filedialog.asksaveasfilename(
            title="Save Agricultural Pathology Report",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if save_path:
            try:
                report_content = self._generate_report_text(self._last_result)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                messagebox.showinfo("Report Exported", f"Pathology report successfully saved to:\n{save_path}")
            except Exception as exc:
                messagebox.showerror("Export Error", f"Failed to save report file:\n{exc}")

    def _on_reset(self):
        """Resets the application canvases and report cards to clean initial state."""
        self.canvas_orig.delete("all")
        self.canvas_heat.delete("all")
        self._current_orig_photo = None
        self._current_overlay_photo = None
        self._current_raw_pil = None
        self._current_image_path = None
        self._last_result = None

        self.lbl_diag_disease.config(text="Ready to Diagnose Leaf Specimen")
        self.lbl_badge_pathogen.config(text="Pathogen: —", bg="#E2E8F0", fg="#334155")
        self.lbl_badge_conf.config(text="Confidence: —", bg="#DBEAFE", fg="#1E40AF")
        self.lbl_badge_severity.config(text="Infected Area: —", bg="#DCFCE7", fg="#166534")

        for item in self.tree_top3.get_children():
            self.tree_top3.delete(item)
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        for txt_widget in (self.txt_symptoms, self.txt_organic, self.txt_chemical, self.txt_prevention, self.txt_full_log):
            txt_widget.delete("1.0", "end")

        self.lbl_status_left.config(text="Ready | Canvases cleared")
        self.lbl_status_right.config(text="Inference: — ms | Engine: Hybrid CV + Random Forest")


def main():
    """Application entry point."""
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
