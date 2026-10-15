"""
Intelligent Spell & Grammar Checker Desktop GUI.

A production-grade Tkinter application designed for University Internal Assessment.
Features:
  - Multi-tier NLP cascade engine integration
  - Background daemon worker threading for zero UI freezes
  - Live character/word metrics and inline error highlighting
  - Interactive Treeview diagnostic inspector with single-click fix application
  - Import/Export file capabilities and sample text demonstration presets
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

from engine import CheckResult, GrammarIssue, SpellGrammarEngine

# UI Palette Constants (Tailwind Slate Theme)
COLOR_BG = "#F8FAFC"          # Slate-50 main window background
COLOR_CARD = "#FFFFFF"        # Pure white panel cards
COLOR_BORDER = "#E2E8F0"      # Slate-200 border
COLOR_TEXT_MAIN = "#0F172A"   # Slate-900 primary text
COLOR_TEXT_MUTED = "#64748B"  # Slate-500 secondary text
COLOR_PRIMARY = "#2563EB"     # Royal Blue 600
COLOR_PRIMARY_HOVER = "#1D4ED8"  # Royal Blue 700
COLOR_SECONDARY_BG = "#F1F5F9"   # Slate-100
COLOR_SECONDARY_FG = "#334155"   # Slate-700
COLOR_SECONDARY_HOVER = "#E2E8F0"

# Tag Highlighting Colors
TAG_SPELL_BG = "#FEE2E2"      # Soft red
TAG_SPELL_FG = "#991B1B"      # Deep red
TAG_GRAMMAR_BG = "#FEF3C7"    # Soft amber
TAG_GRAMMAR_FG = "#92400E"    # Deep amber
TAG_SELECTED_BG = "#BFDBFE"   # Soft blue
TAG_SELECTED_FG = "#1E3A8A"   # Deep blue

# Status Badge Colors
BADGE_INIT_BG = "#FEF3C7"
BADGE_INIT_FG = "#92400E"
BADGE_READY_BG = "#DCFCE7"
BADGE_READY_FG = "#166534"

# Typography
FONT_FAMILY = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
FONT_TITLE = (FONT_FAMILY, 14, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 9)
FONT_HEADING = (FONT_FAMILY, 10, "bold")
FONT_BODY = (FONT_FAMILY, 10)
FONT_BODY_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_CODE = ("Consolas", 10) if sys.platform.startswith("win") else ("Courier", 10)

# Preset Demo Texts for Live Internal Assessment Demonstrations
SAMPLE_TEXTS = [
    (
        "Academic Essay (Grammar, Capitalization & Spelling)",
        (
            "she go to university every day to study computer science. "
            "Defanitely, I has a lot of intresting questions about artifical intelligence. "
            "The the professor explained that an computer must learn from clean data. "
            "However, she don't understand why the algorithm made mistakes."
        ),
    ),
    (
        "Business Email (Articles, Tense & Duplication)",
        (
            "yesterday to buying office supplies for our department. "
            "We recieved a apple and an sandwich from the cafeteria. "
            "The the project manager said they was very pleased with our performance. "
            "Please let me know if you needs any further details."
        ),
    ),
    (
        "Student Lab Report (Spacing, Honorifics & Lexical Typos)",
        (
            "It is an honor to present this experimental research report.  "
            "He have finished the experiment on time.  "
            "There was several obsticales during data collection.  "
            "We must seperate the valid samples before completing the evaluation."
        ),
    ),
]


class SpellGrammarApp:
    """Master Desktop GUI Application for Spell & Grammar Checking."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Intelligent Spell & Grammar Checker - NLP Project")
        self.root.geometry("1050x720")
        self.root.minsize(900, 620)
        self.root.configure(bg=COLOR_BG)

        # Application state
        self.engine: Optional[SpellGrammarEngine] = None
        self.current_result: Optional[CheckResult] = None
        self.current_issues: list[GrammarIssue] = []
        self.selected_issue: Optional[GrammarIssue] = None
        self.sample_index = 0
        self.is_checking = False

        # Protocol binding for graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Build UI layout
        self._setup_styles()
        self._build_header()
        self._build_toolbar()
        self._build_workspace()
        self._build_status_bar()

        # Keyboard shortcuts
        self.root.bind("<Control-Return>", lambda event: self.start_check_thread())

        # Start asynchronous engine initialization on background daemon thread
        # ponytail: daemon thread ensures background initialization never prevents app shutdown
        threading.Thread(target=self._init_engine_worker, daemon=True).start()

    def _setup_styles(self) -> None:
        """Configures ttk widget styling for a clean, modern aesthetic."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT_MAIN, font=FONT_BODY)
        style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_MAIN, font=FONT_BODY)
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_TEXT_MUTED, font=FONT_SMALL)
        style.configure("CardMuted.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_MUTED, font=FONT_SMALL)

        # Treeview styling
        style.configure(
            "Treeview",
            background=COLOR_CARD,
            foreground=COLOR_TEXT_MAIN,
            fieldbackground=COLOR_CARD,
            rowheight=26,
            font=FONT_SMALL,
            borderwidth=1,
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SECONDARY_BG,
            foreground=COLOR_TEXT_MAIN,
            font=FONT_BODY_BOLD,
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", "#1E3A8A")])

        # PanedWindow style
        style.configure("TPanedwindow", background=COLOR_BG)

    def _build_header(self) -> None:
        """Builds top header bar with application branding and live engine status badge."""
        header_frame = tk.Frame(self.root, bg=COLOR_BG, padx=16, pady=10)
        header_frame.pack(fill=tk.X)

        title_container = tk.Frame(header_frame, bg=COLOR_BG)
        title_container.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            title_container,
            text="Intelligent Spell & Grammar Checker",
            font=FONT_TITLE,
            bg=COLOR_BG,
            fg=COLOR_TEXT_MAIN,
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            title_container,
            text="NLP Internal Assessment Project - Multi-Tier Hybrid Engine",
            font=FONT_SUBTITLE,
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED,
        )
        subtitle_lbl.pack(anchor=tk.W)

        # Engine status badge
        self.badge_lbl = tk.Label(
            header_frame,
            text="[Initializing NLP Engine...]",
            font=FONT_BODY_BOLD,
            bg=BADGE_INIT_BG,
            fg=BADGE_INIT_FG,
            padx=12,
            pady=4,
            relief="flat",
        )
        self.badge_lbl.pack(side=tk.RIGHT, pady=4)

    def _build_toolbar(self) -> None:
        """Constructs action toolbar for triggering checks, samples, and file operations."""
        toolbar_container = tk.Frame(self.root, bg=COLOR_CARD, padx=16, pady=8, highlightthickness=1)
        toolbar_container.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)
        toolbar_container.pack(fill=tk.X, padx=16, pady=(0, 10))

        # Primary Action Button
        self.btn_check = tk.Button(
            toolbar_container,
            text="Check & Auto-Correct",
            command=self.start_check_thread,
            bg=COLOR_PRIMARY,
            fg="#FFFFFF",
            activebackground=COLOR_PRIMARY_HOVER,
            activeforeground="#FFFFFF",
            font=FONT_BODY_BOLD,
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
        )
        self.btn_check.pack(side=tk.LEFT, padx=(0, 8))

        # Sample Preset Button
        self.btn_sample = self._create_secondary_button(
            toolbar_container,
            text="Load Sample Text",
            command=self.cycle_sample_text,
        )
        self.btn_sample.pack(side=tk.LEFT, padx=(0, 8))

        # Import File Button
        btn_import = self._create_secondary_button(
            toolbar_container,
            text="Import File",
            command=self.import_text_file,
        )
        btn_import.pack(side=tk.LEFT, padx=(0, 8))

        # Export Corrected File Button
        btn_export = self._create_secondary_button(
            toolbar_container,
            text="Export Corrected",
            command=self.export_text_file,
        )
        btn_export.pack(side=tk.LEFT, padx=(0, 8))

        # Copy Output Button
        btn_copy = self._create_secondary_button(
            toolbar_container,
            text="Copy Output",
            command=self.copy_output_to_clipboard,
        )
        btn_copy.pack(side=tk.LEFT, padx=(0, 8))

        # Clear All Button
        btn_clear = self._create_secondary_button(
            toolbar_container,
            text="Clear All",
            command=self.clear_all,
        )
        btn_clear.pack(side=tk.RIGHT)

    def _create_secondary_button(self, parent: tk.Widget, text: str, command: Any) -> tk.Button:
        """Creates a styled secondary button with hover effects."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLOR_SECONDARY_BG,
            fg=COLOR_SECONDARY_FG,
            activebackground=COLOR_SECONDARY_HOVER,
            activeforeground=COLOR_SECONDARY_FG,
            font=FONT_BODY,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
        )
        btn.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)

        def on_enter(_: tk.Event) -> None:
            btn.config(bg=COLOR_SECONDARY_HOVER)

        def on_leave(_: tk.Event) -> None:
            btn.config(bg=COLOR_SECONDARY_BG)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _build_workspace(self) -> None:
        """Builds 2-column resizable workspace: Text Editors (Left) and Diagnostic Inspector (Right)."""
        workspace_frame = tk.Frame(self.root, bg=COLOR_BG)
        workspace_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        paned = ttk.PanedWindow(workspace_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left Column: Input and Corrected Text Panels
        left_pane = tk.Frame(paned, bg=COLOR_BG)
        paned.add(left_pane, weight=3)

        # 1. Input Text Card
        input_card = tk.Frame(left_pane, bg=COLOR_CARD, padx=12, pady=10, highlightthickness=1)
        input_card.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)
        input_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        input_header = tk.Frame(input_card, bg=COLOR_CARD)
        input_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(input_header, text="Input Text", font=FONT_HEADING, bg=COLOR_CARD, fg=COLOR_TEXT_MAIN).pack(
            side=tk.LEFT
        )
        self.live_counter_lbl = tk.Label(
            input_header, text="Words: 0 | Chars: 0", font=FONT_SMALL, bg=COLOR_CARD, fg=COLOR_TEXT_MUTED
        )
        self.live_counter_lbl.pack(side=tk.RIGHT)

        self.input_text = scrolledtext.ScrolledText(
            input_card,
            wrap=tk.WORD,
            font=FONT_CODE,
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_PRIMARY,
            undo=True,
            padx=8,
            pady=8,
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)

        # Configure highlighting tags
        self.input_text.tag_configure("spelling_err", background=TAG_SPELL_BG, foreground=TAG_SPELL_FG, underline=True)
        self.input_text.tag_configure(
            "grammar_err", background=TAG_GRAMMAR_BG, foreground=TAG_GRAMMAR_FG, underline=True
        )
        self.input_text.tag_configure(
            "selected_err", background=TAG_SELECTED_BG, foreground=TAG_SELECTED_FG, underline=True
        )

        # Bind live metric counter
        self.input_text.bind("<KeyRelease>", lambda event: self.update_live_input_metrics())

        # 2. Corrected Output Card
        output_card = tk.Frame(left_pane, bg=COLOR_CARD, padx=12, pady=10, highlightthickness=1)
        output_card.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)
        output_card.pack(fill=tk.BOTH, expand=True)

        output_header = tk.Frame(output_card, bg=COLOR_CARD)
        output_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(output_header, text="Corrected Output", font=FONT_HEADING, bg=COLOR_CARD, fg=COLOR_TEXT_MAIN).pack(
            side=tk.LEFT
        )
        tk.Label(
            output_header,
            text="Auto-Corrected Clean Text",
            font=FONT_SMALL,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MUTED,
        ).pack(side=tk.RIGHT)

        self.output_text = scrolledtext.ScrolledText(
            output_card,
            wrap=tk.WORD,
            font=FONT_CODE,
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_PRIMARY,
            padx=8,
            pady=8,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Right Column: Diagnostic Inspector and Detail Card
        right_pane = tk.Frame(paned, bg=COLOR_BG)
        paned.add(right_pane, weight=2)

        inspector_card = tk.Frame(right_pane, bg=COLOR_CARD, padx=12, pady=10, highlightthickness=1)
        inspector_card.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)
        inspector_card.pack(fill=tk.BOTH, expand=True)

        inspector_header = tk.Frame(inspector_card, bg=COLOR_CARD)
        inspector_header.pack(fill=tk.X, pady=(0, 6))

        self.issues_header_lbl = tk.Label(
            inspector_header,
            text="Diagnostic Inspector (Issues: 0)",
            font=FONT_HEADING,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MAIN,
        )
        self.issues_header_lbl.pack(side=tk.LEFT)

        # Treeview Diagnostic Table
        tree_container = tk.Frame(inspector_card, bg=COLOR_CARD)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        columns = ("#", "Type", "Error", "Suggestion", "Explanation")
        self.tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("#", text="#")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Error", text="Error")
        self.tree.heading("Suggestion", text="Suggestion")
        self.tree.heading("Explanation", text="Explanation")

        self.tree.column("#", width=32, minwidth=28, anchor=tk.CENTER)
        self.tree.column("Type", width=68, minwidth=60, anchor=tk.CENTER)
        self.tree.column("Error", width=85, minwidth=60)
        self.tree.column("Suggestion", width=95, minwidth=70)
        self.tree.column("Explanation", width=160, minwidth=120)

        scrollbar_y = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self.on_issue_selected)

        # Detailed Issue Inspector Card
        self.detail_frame = tk.LabelFrame(
            inspector_card,
            text=" Detailed Issue Inspector ",
            font=FONT_HEADING,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MAIN,
            padx=10,
            pady=8,
            relief="groove",
        )
        self.detail_frame.pack(fill=tk.X)

        self.detail_rule_lbl = tk.Label(
            self.detail_frame,
            text="Rule ID: None selected",
            font=FONT_SMALL,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MUTED,
            anchor=tk.W,
        )
        self.detail_rule_lbl.pack(fill=tk.X)

        self.detail_msg_lbl = tk.Label(
            self.detail_frame,
            text="Click any issue in the table above to inspect details and apply individual fixes.",
            font=FONT_BODY,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MAIN,
            wraplength=340,
            justify=tk.LEFT,
            anchor=tk.W,
        )
        self.detail_msg_lbl.pack(fill=tk.X, pady=(4, 6))

        detail_action_bar = tk.Frame(self.detail_frame, bg=COLOR_CARD)
        detail_action_bar.pack(fill=tk.X)

        self.btn_apply_single = tk.Button(
            detail_action_bar,
            text="Apply This Single Fix",
            command=self.apply_single_fix,
            bg=COLOR_SECONDARY_BG,
            fg=COLOR_SECONDARY_FG,
            font=FONT_BODY_BOLD,
            relief="flat",
            padx=10,
            pady=4,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.btn_apply_single.pack(side=tk.RIGHT)

    def _build_status_bar(self) -> None:
        """Builds bottom metrics status bar."""
        status_container = tk.Frame(self.root, bg=COLOR_CARD, padx=16, pady=6, highlightthickness=1)
        status_container.config(highlightbackground=COLOR_BORDER, highlightcolor=COLOR_BORDER)
        status_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_metrics_lbl = tk.Label(
            status_container,
            text="Words: 0 | Characters: 0 | Sentences: 0 | Issues Found: 0 (Spelling: 0, Grammar: 0) | Time: 0.0 ms",
            font=FONT_SMALL,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MAIN,
        )
        self.status_metrics_lbl.pack(side=tk.LEFT)

        shortcut_lbl = tk.Label(
            status_container,
            text="Shortcut: Ctrl+Enter (Check Text)",
            font=FONT_SMALL,
            bg=COLOR_CARD,
            fg=COLOR_TEXT_MUTED,
        )
        shortcut_lbl.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Thread-Safe Engine Management & Execution
    # -------------------------------------------------------------------------

    def _init_engine_worker(self) -> None:
        """Background daemon worker for initializing the 3-tier NLP engine."""
        try:
            logger_inst = SpellGrammarEngine()
            # Safely schedule UI update on main thread
            # ponytail: Use root.after(0) to guarantee all widget operations run strictly on Tkinter mainloop
            self.root.after(0, self._on_engine_ready, logger_inst)
        except Exception as e:
            self.root.after(0, self._on_engine_failed, str(e))

    def _on_engine_ready(self, engine_instance: SpellGrammarEngine) -> None:
        """Main-thread callback invoked when engine initialization completes."""
        self.engine = engine_instance
        engine_name = self.engine.active_engine_name
        self.badge_lbl.config(
            text=f"[Engine Ready] {engine_name}",
            bg=BADGE_READY_BG,
            fg=BADGE_READY_FG,
        )

    def _on_engine_failed(self, error_msg: str) -> None:
        """Main-thread callback invoked if all engine tiers fail."""
        self.badge_lbl.config(
            text="[Engine Error]",
            bg=TAG_SPELL_BG,
            fg=TAG_SPELL_FG,
        )
        messagebox.showerror(
            "Engine Error",
            f"Failed to initialize spell/grammar engine:\n{error_msg}\nPlease verify your Python packages.",
        )

    def start_check_thread(self) -> None:
        """Initiates text checking on a background daemon worker to keep the GUI responsive."""
        if self.is_checking:
            return

        if self.engine is None:
            messagebox.showinfo(
                "Initializing",
                "The NLP engine is still initializing in the background. Please try again in a moment.",
            )
            return

        text = self.input_text.get("1.0", tk.END).rstrip("\n")
        if not text.strip():
            messagebox.showwarning("Empty Text", "Please type or paste some text to check.")
            return

        self.is_checking = True
        self.btn_check.config(text="Analyzing...", state=tk.DISABLED)

        # Launch background analysis worker
        threading.Thread(target=self._run_check_worker, args=(text,), daemon=True).start()

    def _run_check_worker(self, text: str) -> None:
        """Background thread worker for NLP analysis."""
        try:
            result = self.engine.analyze(text)
            self.root.after(0, self._display_results, result)
        except Exception as e:
            self.root.after(0, self._on_check_failed, str(e))

    def _on_check_failed(self, err_msg: str) -> None:
        """Handles analysis failure gracefully on the main thread."""
        self.is_checking = False
        self.btn_check.config(text="Check & Auto-Correct", state=tk.NORMAL)
        messagebox.showerror("Analysis Error", f"An error occurred while checking text:\n{err_msg}")

    def _display_results(self, result: CheckResult) -> None:
        """Updates GUI widgets with check results on the main thread."""
        self.is_checking = False
        self.btn_check.config(text="Check & Auto-Correct", state=tk.NORMAL)

        self.current_result = result
        self.current_issues = result.issues

        # 1. Update Corrected Output text area
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", result.corrected_text)

        # 2. Highlight errors inside Input text widget
        self._highlight_errors(result.issues)

        # 3. Populate Diagnostic Inspector Treeview
        self._populate_treeview(result.issues)

        # 4. Update status bar metrics
        spelling_count = sum(1 for i in result.issues if i.category == "Spelling")
        grammar_count = sum(1 for i in result.issues if i.category != "Spelling")
        self.status_metrics_lbl.config(
            text=(
                f"Words: {result.word_count} | Characters: {result.char_count} | "
                f"Sentences: {result.sentence_count} | Issues Found: {len(result.issues)} "
                f"(Spelling: {spelling_count}, Grammar: {grammar_count}) | "
                f"Time: {result.processing_time_ms:.1f} ms"
            )
        )
        self.issues_header_lbl.config(text=f"Diagnostic Inspector (Issues: {len(result.issues)})")

        # Reset selection detail card
        self._reset_detail_card()

    def _highlight_errors(self, issues: list[GrammarIssue]) -> None:
        """Applies inline colored tags for all detected issues in the Input Text widget."""
        # Clear existing highlighting tags
        self.input_text.tag_remove("spelling_err", "1.0", tk.END)
        self.input_text.tag_remove("grammar_err", "1.0", tk.END)
        self.input_text.tag_remove("selected_err", "1.0", tk.END)

        for issue in issues:
            if issue.offset < 0 or issue.length <= 0:
                continue
            # ponytail: Use standard Tkinter index syntax '1.0 + N chars' for accurate character-level tagging
            start_index = f"1.0 + {issue.offset} chars"
            end_index = f"1.0 + {issue.offset + issue.length} chars"
            tag = "spelling_err" if issue.category == "Spelling" else "grammar_err"
            self.input_text.tag_add(tag, start_index, end_index)

    def _populate_treeview(self, issues: list[GrammarIssue]) -> None:
        """Populates the Treeview diagnostic table with detected issues."""
        self.tree.delete(*self.tree.get_children())
        for idx, issue in enumerate(issues, start=1):
            suggestion = issue.replacements[0] if issue.replacements else "N/A"
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx - 1),
                values=(
                    idx,
                    issue.category,
                    issue.error_text or "-",
                    suggestion,
                    issue.message,
                ),
            )

    # -------------------------------------------------------------------------
    # Interactive Diagnostic Inspector Card & Single Fix
    # -------------------------------------------------------------------------

    def on_issue_selected(self, _: tk.Event) -> None:
        """Handles user clicking an issue row in the Treeview table."""
        selected_ids = self.tree.selection()
        if not selected_ids:
            return

        item_idx = int(selected_ids[0])
        if 0 <= item_idx < len(self.current_issues):
            issue = self.current_issues[item_idx]
            self.selected_issue = issue

            # Update detail card labels
            self.detail_rule_lbl.config(
                text=f"Rule: {issue.rule_id} | Category: {issue.category} | Error: '{issue.error_text}'"
            )
            replacements_text = ", ".join(f"'{r}'" for r in issue.replacements) if issue.replacements else "None"
            self.detail_msg_lbl.config(
                text=f"Explanation: {issue.message}\nSuggested Replacement(s): {replacements_text}"
            )

            # Enable single fix button if replacements exist
            if issue.replacements:
                self.btn_apply_single.config(
                    text=f"Apply Fix: '{issue.replacements[0]}'",
                    state=tk.NORMAL,
                    bg=COLOR_PRIMARY,
                    fg="#FFFFFF",
                )
            else:
                self.btn_apply_single.config(text="No Suggested Fix", state=tk.DISABLED, bg=COLOR_SECONDARY_BG)

            # Highlight selected issue in Input Text and scroll to it
            self.input_text.tag_remove("selected_err", "1.0", tk.END)
            start_idx = f"1.0 + {issue.offset} chars"
            end_idx = f"1.0 + {issue.offset + issue.length} chars"
            self.input_text.tag_add("selected_err", start_idx, end_idx)
            self.input_text.see(start_idx)

    def apply_single_fix(self) -> None:
        """Applies suggested replacement for the selected issue directly into the input editor."""
        if not self.selected_issue or not self.selected_issue.replacements:
            return

        replacement = self.selected_issue.replacements[0]
        offset = self.selected_issue.offset
        length = self.selected_issue.length

        start_idx = f"1.0 + {offset} chars"
        end_idx = f"1.0 + {offset + length} chars"

        # Apply replacement directly in text widget
        self.input_text.delete(start_idx, end_idx)
        self.input_text.insert(start_idx, replacement)

        # Trigger re-analysis to refresh all highlights and offsets
        self.start_check_thread()

    def _reset_detail_card(self) -> None:
        """Resets the inspector detail card to its initial state."""
        self.selected_issue = None
        self.detail_rule_lbl.config(text="Rule ID: None selected")
        self.detail_msg_lbl.config(
            text="Click any issue in the table above to inspect details and apply individual fixes."
        )
        self.btn_apply_single.config(
            text="Apply This Single Fix",
            state=tk.DISABLED,
            bg=COLOR_SECONDARY_BG,
            fg=COLOR_SECONDARY_FG,
        )

    # -------------------------------------------------------------------------
    # Toolbar Actions: Sample Texts, File I/O, Clipboard
    # -------------------------------------------------------------------------

    def cycle_sample_text(self) -> None:
        """Cycles through pre-configured realistic sample texts with diverse errors."""
        name, text = SAMPLE_TEXTS[self.sample_index]
        self.sample_index = (self.sample_index + 1) % len(SAMPLE_TEXTS)

        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", text)
        self.update_live_input_metrics()

        # Automatically trigger check on sample load
        self.start_check_thread()

    def import_text_file(self) -> None:
        """Imports text from a local .txt or .md file."""
        file_path = filedialog.askopenfilename(
            title="Import Document",
            filetypes=[("Text and Markdown Files", "*.txt;*.md"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", content)
            self.update_live_input_metrics()
            self.start_check_thread()
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not read file:\n{e}")

    def export_text_file(self) -> None:
        """Exports auto-corrected text to a user-specified text file."""
        corrected_content = self.output_text.get("1.0", tk.END).rstrip("\n")
        if not corrected_content.strip():
            messagebox.showwarning("Export Warning", "Corrected output is empty. Nothing to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Corrected Document",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("Markdown File", "*.md"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(corrected_content)
            messagebox.showinfo("Export Successful", f"File saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write file:\n{e}")

    def copy_output_to_clipboard(self) -> None:
        """Copies the auto-corrected output to system clipboard."""
        corrected_content = self.output_text.get("1.0", tk.END).rstrip("\n")
        if not corrected_content.strip():
            messagebox.showwarning("Clipboard", "Corrected output is empty.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(corrected_content)

        # Temporary confirmation in status metrics bar
        original_status = self.status_metrics_lbl.cget("text")
        self.status_metrics_lbl.config(text="[Success] Corrected text copied to clipboard!")
        self.root.after(2000, lambda: self.status_metrics_lbl.config(text=original_status))

    def clear_all(self) -> None:
        """Clears both text areas, highlights, inspector table, and resets status metrics."""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.input_text.tag_remove("spelling_err", "1.0", tk.END)
        self.input_text.tag_remove("grammar_err", "1.0", tk.END)
        self.input_text.tag_remove("selected_err", "1.0", tk.END)
        self.tree.delete(*self.tree.get_children())
        self._reset_detail_card()
        self.update_live_input_metrics()
        self.issues_header_lbl.config(text="Diagnostic Inspector (Issues: 0)")
        self.status_metrics_lbl.config(
            text="Words: 0 | Characters: 0 | Sentences: 0 | Issues Found: 0 (Spelling: 0, Grammar: 0) | Time: 0.0 ms"
        )

    def update_live_input_metrics(self) -> None:
        """Updates live word and character counts on key typing."""
        text = self.input_text.get("1.0", tk.END).rstrip("\n")
        chars = len(text)
        words = len(text.split()) if text.strip() else 0
        self.live_counter_lbl.config(text=f"Words: {words} | Chars: {chars}")

    def on_closing(self) -> None:
        """Performs safe cleanup of background engine before destroying root window."""
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


def main() -> None:
    """Main application launcher."""
    root = tk.Tk()
    app = SpellGrammarApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
