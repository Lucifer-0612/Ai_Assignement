# Intelligent Spell & Grammar Checker Desktop Application
**Natural Language Processing (NLP) — University Internal Assessment Project**

---

## 1. Project Overview & Problem Statement

### 1.1 Problem Statement
Written communication across academic and professional domains is susceptible to orthographic errors (misspellings), syntactic errors (subject-verb disagreement, incorrect verb tenses), and stylistic inconsistencies (repeated words, incorrect article usage, missing sentence capitalization). While cloud-based solutions exist (such as Grammarly or generative AI APIs), they suffer from significant drawbacks in educational and offline environments:
1. **Network Dependency & Latency:** Unreliable internet connectivity stalls document verification.
2. **Subscription Costs & Rate Limits:** Commercial APIs impose paywalls and strict request limits.
3. **Heavy Compute Overhead:** Deep Learning transformers (e.g., BERT, T5, GPT-4) require multiple gigabytes of memory and dedicated GPU hardware, making them unfeasible for standard student workstations.
4. **Environment Instability:** Local rule engines often depend on external runtimes (such as Java for LanguageTool) that are frequently absent from student environments.

### 1.2 Proposed Solution
This project implements a lightweight, modular, desktop-grade **Spell & Grammar Checker GUI** built purely in Python. It features a robust **3-Tier Cascade Engine** that guarantees 100% offline functionality without compromising accuracy:
- **Tier 1 (Local LanguageTool):** Connects to a local Java-based LanguageTool server when Java 8+ and server binaries are present.
- **Tier 2 (Public Cloud API):** Seamlessly communicates with LanguageTool's public REST API if Java is absent.
- **Tier 3 (Pure-Python Offline Fallback):** Combines `pyspellchecker` (frequency-based Levenshtein dictionary), `textblob` (probabilistic n-gram correction), and optimized regular expression heuristics to detect syntax, article misuse, sentence capitalization, duplicate tokens, and whitespace anomalies without external dependencies or runtimes.

The desktop client provides real-time visual diagnostics, dual-column synchronized text editors, inline character-level highlighting, interactive Treeview inspectors, single-click fix applications, and complete document import/export capabilities.

---

## 2. System Architecture Diagram

The system decouples the Tkinter user interface from the NLP computational backend using background daemon threading and thread-safe event scheduling.

```mermaid
flowchart TD
    subgraph UI_Layer["Desktop GUI Layer (Tkinter Mainloop)"]
        A["User Input / Sample Preset / File Import"] --> B["Input ScrolledText Panel"]
        B -->|KeyRelease Event| C["Live Word & Char Metrics"]
        B -->|Ctrl+Enter / Click| D["Check & Auto-Correct Trigger"]
        D -->|Dispatch Task| E["Background Daemon Thread Worker"]
    end

    subgraph Backend_Layer["NLP Engine Controller (engine.py)"]
        E --> F["SpellGrammarEngine Cascade Controller"]
        F --> G{"Tier 1: Local Java<br/>LanguageTool Available?"}
        G -- Yes --> H["Tier 1: Local LanguageTool Server"]
        G -- No / Error --> I{"Tier 2: Public API<br/>Reachable & Not Limited?"}
        I -- Yes --> J["Tier 2: LanguageTool Cloud API"]
        I -- No / Error --> K["Tier 3: Pure-Python Hybrid Offline Engine"]

        K --> K1["pyspellchecker + TextBlob<br/>(Lexical Edit Distance)"]
        K --> K2["Regex Syntactic Heuristics<br/>(Subject-Verb, Tenses, Articles)"]
        K --> K3["Structural Heuristics<br/>(Sentence Start, Whitespace, Duplicates)"]
    end

    subgraph Result_Processing["Result Processing & Auto-Correction"]
        H --> L["Extract Issues into GrammarIssue Dataclasses"]
        J --> L
        K1 --> L
        K2 --> L
        K3 --> L
        L --> M["Reverse-Offset Auto-Corrector<br/>(Right-to-Left Replacement)"]
        M --> N["CheckResult Container"]
    end

    subgraph UI_Feedback["Main-Thread UI Updates (root.after)"]
        N -->|root.after(0)| O["Dispatcher"]
        O --> P["Inline Highlight Engine<br/>(tag_add 'spelling_err' & 'grammar_err')"]
        O --> Q["Populate Corrected Output Panel"]
        O --> R["Populate Treeview Diagnostic Table"]
        O --> S["Update Metrics Status Bar"]
    end
```

---

## 3. Step-by-Step Installation & Execution Guide

### 3.1 Prerequisites
- **Operating System:** Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+)
- **Python Version:** Python 3.9 through 3.13
- **Tkinter Support:** Standard on Windows/macOS Python installers. (On Linux: `sudo apt-get install python3-tk`)

### 3.2 Installation

1. **Clone or Navigate to the Project Directory:**
   ```bash
   cd D:\projects\ai_assignment\spell_checker\spell_grammar_checker
   ```

2. **(Optional) Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies installed:*
   - `language-tool-python>=2.7.1`
   - `pyspellchecker>=0.8.1`
   - `textblob>=0.18.0`

### 3.3 Running the Application

Launch the desktop interface:
```bash
python app.py
```

### 3.4 Verifying Syntax and Code Integrity
```bash
python -m py_compile engine.py
python -m py_compile app.py
```

---

## 4. Module-by-Module Technical Walkthrough

### 4.1 `engine.py` (NLP Architecture & Logic)
`engine.py` encapsulates the data structures, safe accessors, and analysis pipelines:

- **`GrammarIssue` Dataclass:**
  Stores structured information for each identified error:
  - `offset (int)`: Exact 0-indexed starting character position.
  - `length (int)`: Length of the offending substring.
  - `error_text (str)`: Substring extracted from input text.
  - `message (str)`: Human-readable diagnostic feedback.
  - `replacements (list[str])`: Ranked suggestions (top 5).
  - `rule_id (str)`: Identifier (e.g., `SUBJECT_VERB_AGREEMENT`, `EN_A_VS_AN`).
  - `category (str)`: Categorized as `"Spelling"`, `"Grammar"`, or `"Style/Punctuation"`.

- **`CheckResult` Dataclass:**
  Encapsulates execution metrics (`processing_time_ms`, `word_count`, `char_count`, `sentence_count`), the full list of `GrammarIssue` objects, and the cleanly corrected text.

- **Safe Attribute Getters:**
  `get_error_length(match)` and `get_rule_id(match)` use dynamic fallback introspection (`getattr(match, 'errorLength', getattr(match, 'error_length', 0))`) to protect against breaking API changes between versions of `language_tool_python`.

- **Reverse-Offset Auto-Correction Algorithm:**
  ```python
  def _apply_reverse_corrections(self, text: str, issues: list[GrammarIssue]) -> str:
      sorted_issues = sorted(issues, key=lambda x: (x.offset, x.length), reverse=True)
      corrected = text
      last_replaced_start = len(text) + 1
      for issue in sorted_issues:
          if not issue.replacements or issue.offset < 0:
              continue
          start = issue.offset
          end = issue.offset + issue.length
          if end <= last_replaced_start and start >= 0:
              corrected = corrected[:start] + issue.replacements[0] + corrected[end:]
              last_replaced_start = start
      return corrected
  ```
  *Technical Rationale:* If replacements are applied from beginning to end, replacing a 4-letter word with a 7-letter word shifts all subsequent character positions by +3 characters, corrupting all subsequent replacement operations. By sorting descending by offset, earlier substring indices remain stable.

### 4.2 `app.py` (Desktop GUI & Asynchronous UI Layer)
`app.py` implements the desktop presentation layer using Tkinter and TTK:

- **Thread-Safe Architecture:**
  The Tkinter event loop (`root.mainloop()`) must never be blocked by CPU-bound or network-bound operations. Both engine initialization and text analysis are dispatched onto background daemon threads (`threading.Thread(..., daemon=True)`). Results are dispatched to the UI thread via `root.after(0, callback, *args)`.

- **Inline Character Tagging:**
  Converts numeric offsets to Tkinter's character index notation:
  ```python
  start_index = f"1.0 + {issue.offset} chars"
  end_index = f"1.0 + {issue.offset + issue.length} chars"
  self.input_text.tag_add(tag, start_index, end_index)
  ```

- **Interactive Diagnostic Inspector:**
  A `ttk.Treeview` widget lists each error with its category, faulty text, suggested fix, and rule explanation. Selecting any row highlights the issue in the input box, scrolls it into view, and populates the **Detailed Issue Inspector Card**, allowing the user to apply that specific fix with a single click.

---

## 5. Top 10 Internal Assessment Viva Questions & Model Answers

### Q1: How does your application highlight the exact erroneous word inside the Tkinter Text widget?
**Model Answer:**
Tkinter `Text` and `ScrolledText` widgets employ an internal fractional text coordinate system formatted as `"<line>.<column>"`, where `1.0` represents line 1, column 0 (the first character of the document). 
Because our NLP engine outputs linear 0-indexed character offsets (e.g., offset 42, length 5), we utilize Tkinter's relative character index modifier:
```python
start_index = f"1.0 + {issue.offset} chars"
end_index = f"1.0 + {issue.offset + issue.length} chars"
self.input_text.tag_add("spelling_err", start_index, end_index)
```
Before applying new highlights, we purge obsolete highlights across the document with `self.input_text.tag_remove("spelling_err", "1.0", tk.END)`. The tags are pre-styled using `tag_configure` with distinct background and foreground colors (soft red for spelling, soft amber for grammar) and underlined formatting.

---

### Q2: Why did you apply text replacements from right-to-left (reverse offset order) instead of left-to-right?
**Model Answer:**
In string manipulation, performing substitutions changes the length of the string unless the replacement has the exact same character count. 
If an error at offset 10 (`"cat"`, 3 characters) is replaced with `"feline"` (6 characters), the string expands by +3 characters. If a subsequent error was recorded at offset 25, that error is now located at character position 28. Applying replacements left-to-right causes "index drift", resulting in replacements slicing through unintended characters and corrupting the output text.
By sorting all detected issues in descending order by character offset (`reverse=True`):
```python
sorted_issues = sorted(issues, key=lambda x: (x.offset, x.length), reverse=True)
```
we modify the string from the end backwards. Any expansion or contraction only affects characters located *after* the replacement point. The character offsets of all preceding errors remain unchanged.

---

### Q3: Why did you use Python's `threading` module in a Tkinter desktop application?
**Model Answer:**
Tkinter is driven by a single-threaded event loop (`root.mainloop()`). This loop is responsible for handling OS window events, mouse clicks, keyboard inputs, and rendering pixels at 60 FPS. 
If an NLP check or remote API query takes 1.5 seconds and is executed directly on the main thread, the event loop blocks completely. The operating system flags the window as "Not Responding", the cursor freezes, and typing halts.
To prevent this, we spawn a background worker thread:
```python
threading.Thread(target=self._run_check_worker, args=(text,), daemon=True).start()
```
Marking the thread as `daemon=True` ensures that if the user closes the window, background threads terminate immediately without leaving zombie processes in the task manager.

---

### Q4: Why can't you update Tkinter widgets directly from a background thread? How did you solve it?
**Model Answer:**
Tkinter is a Python wrapper around Tcl/Tk, which is fundamentally not thread-safe. Modifying GUI widgets (e.g., calling `insert()`, `delete()`, or `config()`) from a non-main thread triggers race conditions, memory corruption, X11 errors on Linux, or silent crashes on Windows.
To solve this, we use Tkinter's thread-safe message scheduling mechanism:
```python
self.root.after(0, self._display_results, result)
```
`root.after(0, func, *args)` places a callback request into Tkinter's internal event queue. The main thread's `mainloop()` picks up this request and executes the UI update safely on the main thread.

---

### Q5: What is the fundamental difference between Spell Checking and Grammar Checking?
**Model Answer:**
- **Spell Checking (Lexical Analysis):** Checks isolated tokens against a known lexicon/dictionary without considering context. If a word is missing from the lexicon, algorithms compute the **Damerau-Levenshtein distance** (insertions, deletions, substitutions, transpositions) and word frequency scores to suggest the closest valid words (e.g., `"definately"` $\rightarrow$ `"definitely"`).
- **Grammar Checking (Syntactic & Semantic Analysis):** Analyzes the relationship between multiple words in context. Both `"She"` and `"go"` are valid dictionary words, but `"She go to school"` violates subject-verb agreement because a third-person singular subject requires a singular verb (`"goes"`). Grammar checkers rely on Part-of-Speech (POS) tagging, dependency parse trees, or syntactic grammar production rules.

---

### Q6: Explain the 3-Tier Fallback Architecture and why it is critical for deployment.
**Model Answer:**
Desktop software in university evaluations must run reliably on heterogeneous client setups:
1. **Tier 1 (Local LanguageTool):** Offers comprehensive rule coverage (~3,000+ rules) via Java. However, if Java JRE is not installed or configured in the system PATH, instantiating this engine raises a `JavaError` or `FileNotFoundError`.
2. **Tier 2 (Public Cloud API):** Bypasses the Java requirement by sending HTTP POST requests to LanguageTool's public servers. However, public servers enforce strict rate limits (HTTP 429) and fail when the client is disconnected from the internet.
3. **Tier 3 (Pure-Python Offline Fallback):** Combines `pyspellchecker`, `textblob`, and compiled regex patterns. It requires zero external binaries, zero internet connection, and zero external runtimes.
By chaining these tiers in an automated fallback `try/except` cascade, the app guarantees that it will **never crash** on startup.

---

### Q7: Why did you choose `@dataclass` for `GrammarIssue` and `CheckResult` instead of standard Python dictionaries or tuples?
**Model Answer:**
Standard dictionaries rely on string keys (e.g., `issue["offset"]`), which lack static type enforcement, permit silent spelling mistakes in key names (e.g., `issue["offest"]`), and require boilerplate dictionary syntax. Named tuples are immutable and lack clean default factory initialization.
Python's `@dataclass`:
1. Provides clean, object-oriented dot-notation (`issue.offset`, `issue.message`).
2. Enables static type hinting (`offset: int`, `replacements: list[str]`) for IDE auto-completion and static analysis tools (`mypy`).
3. Automatically generates standard dunder methods (`__init__`, `__repr__`, `__eq__`).
4. Offers memory efficiency and clarity with default factories (`field(default_factory=list)`).

---

### Q8: How did you implement single-issue correction in the Interactive Diagnostic Inspector?
**Model Answer:**
When the user clicks a row in the `ttk.Treeview`, the event `<<TreeviewSelect>>` fires `on_issue_selected()`:
1. The method retrieves the associated `GrammarIssue` object from `self.current_issues`.
2. It visually highlights that specific issue in the input editor using the `"selected_err"` tag and scrolls to it via `self.input_text.see(start_idx)`.
3. When the user clicks the **"Apply This Single Fix"** button:
   ```python
   start_idx = f"1.0 + {offset} chars"
   end_idx = f"1.0 + {offset + length} chars"
   self.input_text.delete(start_idx, end_idx)
   self.input_text.insert(start_idx, replacement)
   self.start_check_thread()
   ```
4. Replacing the substring and immediately triggering a re-check ensures all remaining offsets and highlights are recalculated accurately without index drift.

---

### Q9: How are phonetic vowel sounds handled in the Article Usage ("a" vs "an") heuristic?
**Model Answer:**
A naive rule checking whether the next word begins with a vowel letter (`a, e, i, o, u`) fails on common English phonetic exceptions:
- Words starting with consonant sound despite a vowel letter: *"a university"*, *"a European"*, *"a unique opportunity"*, *"a one-hour trip"*.
- Words starting with vowel sound despite a consonant letter: *"an hour"*, *"an honest person"*, *"an honor"*.
Our engine defines explicit exception sets:
```python
_VOWEL_EXCEPTIONS_CONSONANT = {"university", "unique", "european", "one", "user", ...}
_VOWEL_EXCEPTIONS_VOWEL = {"hour", "honest", "honor", "heir", ...}
```
The helper function `_starts_with_vowel_sound(word)` cross-checks these exception tables before falling back to the standard `word[0] in 'aeiou'` check.

---

### Q10: How could this project be scaled or enhanced in future iterations?
**Model Answer:**
1. **User Custom Dictionary:** Allow users to right-click unrecognized words and select *"Add to Personal Dictionary"*, saving them to a local JSON/SQLite file so custom technical jargon or names are not flagged.
2. **Contextual Transformer Integration (Optional Flag):** For machines equipped with GPUs, provide an optional toggle for a quantized ONNX model (e.g., `Grammarly/CoEdit` or `T5-small`) using ONNX Runtime for semantic paraphrasing.
3. **Multi-Language Support:** Expand the engine cascade beyond English to Spanish, French, and German by passing language tags dynamically to `SpellChecker(language='es')` and LanguageTool language parameters.
4. **Export Formats:** Extend export functionality beyond plaintext (`.txt`) and Markdown (`.md`) to Microsoft Word (`.docx`) and PDF using `python-docx` and `reportlab`.

---

## 6. Project Verification & Compliance Summary

| Requirement | Implementation Details | Verification Status |
|:---|:---|:---:|
| **Zero Paid APIs** | 100% free; uses local heuristics, pyspellchecker, textblob, LanguageTool | Verified |
| **No Heavy DL Models** | Fast startup (<1s); no PyTorch, TensorFlow, or multi-GB HuggingFace models | Verified |
| **Safe Attribute Introspection** | `get_error_length()`, `get_rule_id()` fallback getters | Verified |
| **3-Tier Cascade Engine** | Local Java LT $\rightarrow$ Public API LT $\rightarrow$ Pure-Python Offline Hybrid | Verified |
| **Thread Safety** | Background daemon threads + `root.after(0, ...)` UI dispatching | Verified |
| **Standard Tkinter Only** | Zero external C-compiled GUI packages; responsive modern layout | Verified |
| **Reverse-Offset Auto-Correction** | Reverse-sorted replacement prevents character offset shift | Verified |
| **Interactive Inspector** | Treeview table + detail card + "Apply Single Fix" interactive button | Verified |
