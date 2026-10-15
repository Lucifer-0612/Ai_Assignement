"""
NLP Engine Module for Desktop Spell & Grammar Checker.

Provides a 3-tier fallback architecture:
  Tier 1: Local LanguageTool (Java-based en-US server)
  Tier 2: LanguageTool Public API (Remote Cloud Service)
  Tier 3: Pure-Python Offline Fallback (pyspellchecker + TextBlob + Regex Heuristics)

Designed with safe attribute accessors, robust error handling, and reverse-offset
auto-correction to prevent index corruption.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

# Safe optional imports for resilient fallback execution
try:
    import language_tool_python
    from language_tool_python.download_lt import LocalLanguageTool, LTP_DOWNLOAD_VERSION
except ImportError:
    language_tool_python = None
    LocalLanguageTool = None
    LTP_DOWNLOAD_VERSION = None

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

try:
    from textblob import TextBlob, Word
except ImportError:
    TextBlob = None
    Word = None

logger = logging.getLogger(__name__)


@dataclass
class GrammarIssue:
    """Represents a single spelling, grammar, or style issue detected in text."""

    offset: int
    length: int
    error_text: str
    message: str
    replacements: list[str] = field(default_factory=list)
    rule_id: str = "UNKNOWN"
    category: str = "Grammar"


@dataclass
class CheckResult:
    """Encapsulates full check results, metrics, and auto-corrected text."""

    original_text: str
    corrected_text: str
    issues: list[GrammarIssue] = field(default_factory=list)
    engine_used: str = "None"
    processing_time_ms: float = 0.0
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0


# Safe attribute getters to avoid AttributeError across language_tool_python versions
def get_error_length(match: Any) -> int:
    """Safely extract error length from LanguageTool match object."""
    # ponytail: language_tool_python uses either errorLength or error_length depending on version
    return int(getattr(match, "errorLength", getattr(match, "error_length", 0)))


def get_rule_id(match: Any) -> str:
    """Safely extract rule ID from LanguageTool match object."""
    return str(getattr(match, "ruleId", getattr(match, "rule_id", "UNKNOWN")))


def _categorize_rule(rule_id: str, rule_issue_type: str = "", category: str = "") -> str:
    """Classifies an issue into Spelling, Style/Punctuation, or Grammar."""
    r_id = rule_id.lower()
    combined = f"{r_id} {rule_issue_type.lower()} {category.lower()}"
    if any(k in combined for k in ["morfologik", "spell", "typo", "misspelling"]):
        return "Spelling"
    if any(k in combined for k in ["space", "whitespace", "punctuation", "style", "casing", "uppercase", "comma"]):
        return "Style/Punctuation"
    return "Grammar"


# Phonetic vowel detection for article heuristics (a vs an)
_VOWEL_EXCEPTIONS_CONSONANT = {
    "university", "universe", "uniform", "unique", "union", "unit",
    "european", "euphemism", "user", "useful", "one", "once", "ufo"
}
_VOWEL_EXCEPTIONS_VOWEL = {
    "hour", "hours", "hourly", "honest", "honesty", "honor", "honour", "heir", "heiress"
}


def _starts_with_vowel_sound(word: str) -> bool:
    """Determine whether an English word begins with a vowel sound."""
    cleaned = word.strip().lower()
    if not cleaned:
        return False
    if cleaned in _VOWEL_EXCEPTIONS_CONSONANT:
        return False
    if cleaned in _VOWEL_EXCEPTIONS_VOWEL:
        return True
    return cleaned[0] in "aeiou"


class SpellGrammarEngine:
    """
    3-Tier cascade Spell & Grammar Checking Engine.
    Guarantees operation even on systems without Java or active internet.
    """

    def __init__(self, preferred_lang: str = "en-US") -> None:
        self.preferred_lang = preferred_lang
        self.tool: Any = None
        self.spell: Any = None
        self.mode = "uninitialized"
        self.active_engine_name = "Initializing..."
        self._init_cascade()

    def _init_cascade(self) -> None:
        """Tries Tier 1 (local Java LanguageTool) -> Tier 2 (Public API) -> Tier 3 (Offline fallback)."""
        # Tier 1: Local LanguageTool (requires Java and pre-installed server binaries)
        if language_tool_python is not None:
            try:
                # ponytail: Verify Java on PATH and avoid blocking on 260MB download if uninstalled
                has_java = shutil.which("java") is not None
                if not has_java:
                    raise RuntimeError("Java is not found in PATH for local LanguageTool.")

                is_installed = False
                if LocalLanguageTool is not None and LTP_DOWNLOAD_VERSION is not None:
                    try:
                        helper = LocalLanguageTool.from_version_name(LTP_DOWNLOAD_VERSION)
                        is_installed = bool(helper.get_installed_versions())
                    except Exception:
                        is_installed = False

                allow_download = os.environ.get("ALLOW_LT_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes")
                if not is_installed and not allow_download:
                    raise RuntimeError("Local LanguageTool server binaries are not pre-downloaded.")

                logger.info("Attempting Tier 1: Local LanguageTool (%s)...", self.preferred_lang)
                self.tool = language_tool_python.LanguageTool(self.preferred_lang)
                self.active_engine_name = f"LanguageTool Local ({self.preferred_lang})"
                self.mode = "tier1_local"
                logger.info("Tier 1 initialized successfully.")
                return
            except Exception as e:
                logger.warning("Tier 1 unavailable (%s). Cascading to Tier 2...", e)

            # Tier 2: Remote Public API LanguageTool (requires Internet, no Java needed)
            try:
                logger.info("Attempting Tier 2: Public API LanguageTool (%s)...", self.preferred_lang)
                self.tool = language_tool_python.LanguageToolPublicAPI(self.preferred_lang)
                # Quick verification query to catch rate limits immediately
                self.tool.check("test")
                self.active_engine_name = f"LanguageTool Public API ({self.preferred_lang})"
                self.mode = "tier2_public_api"
                logger.info("Tier 2 initialized successfully.")
                return
            except Exception as e:
                logger.warning("Tier 2 unavailable (%s). Cascading to Tier 3...", e)
                if self.tool is not None:
                    try:
                        self.tool.close()
                    except Exception:
                        pass
                self.tool = None

        # Tier 3: Pure-Python Offline Fallback (Zero external server / Java dependency)
        logger.info("Engaging Tier 3: Pure-Python Offline Fallback...")
        self.tool = None
        if SpellChecker is not None:
            try:
                self.spell = SpellChecker(language="en")
            except Exception as e:
                logger.warning("pyspellchecker initialization error: %s", e)
                self.spell = None
        self.active_engine_name = "Offline Hybrid NLP (Pure-Python)"
        self.mode = "tier3_offline"

    def analyze(self, text: str) -> CheckResult:
        """Analyzes text for errors, applies reverse-offset auto-correction, and returns metrics."""
        t_start = time.perf_counter()

        # Count text statistics using standard regex
        # ponytail: standard library re is faster and cleaner than extra tokenizer libraries
        words = re.findall(r"\b\w+(?:'\w+)?\b", text)
        word_count = len(words)
        char_count = len(text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = len(sentences) if sentences else (1 if text.strip() else 0)

        if not text.strip():
            return CheckResult(
                original_text=text,
                corrected_text=text,
                issues=[],
                engine_used=self.active_engine_name,
                processing_time_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                word_count=0,
                char_count=char_count,
                sentence_count=0,
            )

        if self.mode in ("tier1_local", "tier2_public_api") and self.tool is not None:
            issues = self._check_with_language_tool(text)
        else:
            issues = self._check_with_offline_fallback(text)

        # Reverse-offset auto-correction (replacing right-to-left prevents index drift)
        corrected_text = self._apply_reverse_corrections(text, issues)

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        return CheckResult(
            original_text=text,
            corrected_text=corrected_text,
            issues=issues,
            engine_used=self.active_engine_name,
            processing_time_ms=round(duration_ms, 2),
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
        )

    def _check_with_language_tool(self, text: str) -> list[GrammarIssue]:
        """Runs checks via language_tool_python instance."""
        issues: list[GrammarIssue] = []
        try:
            matches = self.tool.check(text)
            for m in matches:
                offset = int(getattr(m, "offset", 0))
                length = get_error_length(m)
                rule_id = get_rule_id(m)
                message = str(getattr(m, "message", "Grammar/spelling issue detected."))
                replacements = list(getattr(m, "replacements", []))[:5]
                raw_cat = str(getattr(m, "category", ""))
                rule_type = str(getattr(m, "ruleIssueType", ""))
                category = _categorize_rule(rule_id, rule_type, raw_cat)

                # Extract exact erroneous substring safely
                error_text = text[offset : offset + length] if 0 <= offset <= len(text) else ""

                issues.append(
                    GrammarIssue(
                        offset=offset,
                        length=length,
                        error_text=error_text,
                        message=message,
                        replacements=replacements,
                        rule_id=rule_id,
                        category=category,
                    )
                )
        except Exception as e:
            logger.error("Error during LanguageTool check: %s. Falling back to offline analyzer.", e)
            return self._check_with_offline_fallback(text)

        return issues

    def _check_with_offline_fallback(self, text: str) -> list[GrammarIssue]:
        """
        Pure-Python fallback combining pyspellchecker, TextBlob, and regex heuristics:
          1. Misspelled words (pyspellchecker + TextBlob)
          2. Repeated consecutive words ('the the')
          3. Article usage ('a' vs 'an')
          4. Sentence start capitalization
          5. Consecutive spaces
          6. Common subject-verb agreement and tense errors
        """
        issues: list[GrammarIssue] = []

        # 1. Spelling verification
        if self.spell is not None:
            # ponytail: Use finditer to locate exact word character offsets in one pass
            for match in re.finditer(r"\b[A-Za-z']+\b", text):
                word = match.group(0)
                # Ignore single letters except 'a' and 'I'
                if len(word) == 1 and word.lower() not in ("a", "i"):
                    continue
                if len(word) > 1 and word.lower() in self.spell.unknown([word.lower()]):
                    candidates: list[str] = []

                    # Prioritize TextBlob spellcheck probability ranking when available
                    if Word is not None:
                        try:
                            tb_scores = Word(word.lower()).spellcheck()
                            candidates = [c for c, score in tb_scores if score > 0.05][:5]
                        except Exception:
                            candidates = []

                    # Supplement with pyspellchecker candidates
                    if not candidates:
                        candidates = list(self.spell.candidates(word.lower()))[:5]

                    # Preserve original capitalization if input word was Titlecase or Uppercase
                    if word.istitle():
                        candidates = [c.capitalize() for c in candidates]
                    elif word.isupper():
                        candidates = [c.upper() for c in candidates]

                    issues.append(
                        GrammarIssue(
                            offset=match.start(),
                            length=len(word),
                            error_text=word,
                            message=f"Possible spelling mistake. Suggested corrections: {', '.join(candidates[:3]) if candidates else 'None'}",
                            replacements=candidates,
                            rule_id="SPELL_CHECK_OFFLINE",
                            category="Spelling",
                        )
                    )

        # 2. Repeated consecutive words (e.g. 'the the', 'is is')
        for match in re.finditer(r"\b([A-Za-z]+)\s+\1\b", text, flags=re.IGNORECASE):
            full_match = match.group(0)
            single_word = match.group(1)
            issues.append(
                GrammarIssue(
                    offset=match.start(),
                    length=len(full_match),
                    error_text=full_match,
                    message=f"Repeated consecutive word '{single_word}'.",
                    replacements=[single_word],
                    rule_id="REPEATED_WORD",
                    category="Grammar",
                )
            )

        # 3. Article misuse ('a apple' -> 'an apple', 'an book' -> 'a book')
        for match in re.finditer(r"\b(a|an)\s+([A-Za-z]+)\b", text, flags=re.IGNORECASE):
            art = match.group(1)
            nxt = match.group(2)
            has_vowel_sound = _starts_with_vowel_sound(nxt)

            if art.lower() == "a" and has_vowel_sound:
                correct_art = "An" if art.istitle() else "an"
                issues.append(
                    GrammarIssue(
                        offset=match.start(1),
                        length=len(art),
                        error_text=art,
                        message=f"Use '{correct_art}' instead of '{art}' before vowel sound '{nxt}'.",
                        replacements=[correct_art],
                        rule_id="EN_A_VS_AN",
                        category="Grammar",
                    )
                )
            elif art.lower() == "an" and not has_vowel_sound:
                correct_art = "A" if art.istitle() else "a"
                issues.append(
                    GrammarIssue(
                        offset=match.start(1),
                        length=len(art),
                        error_text=art,
                        message=f"Use '{correct_art}' instead of '{art}' before consonant sound '{nxt}'.",
                        replacements=[correct_art],
                        rule_id="EN_A_VS_AN",
                        category="Grammar",
                    )
                )

        # 4. Missing sentence capitalization
        for match in re.finditer(r"(?:^|[.!?]\s+)([a-z])", text):
            char_offset = match.start(1)
            char = match.group(1)
            issues.append(
                GrammarIssue(
                    offset=char_offset,
                    length=1,
                    error_text=char,
                    message="Sentences should start with an uppercase letter.",
                    replacements=[char.upper()],
                    rule_id="UPPERCASE_SENTENCE_START",
                    category="Style/Punctuation",
                )
            )

        # 5. Multiple consecutive spaces
        for match in re.finditer(r"[ ]{2,}", text):
            issues.append(
                GrammarIssue(
                    offset=match.start(),
                    length=len(match.group(0)),
                    error_text=match.group(0),
                    message="Multiple consecutive spaces detected.",
                    replacements=[" "],
                    rule_id="WHITESPACE_MULTIPLE_SPACES",
                    category="Style/Punctuation",
                )
            )

        # 6. Common subject-verb agreement and tense errors
        # ponytail: Compile regex rules into a lookup table for O(N) zero-overhead pattern matching
        grammar_patterns = [
            (
                r"\b(She|He|It)\s+(go)\b",
                "goes",
                "Subject-verb agreement: third-person singular requires 'goes'.",
            ),
            (
                r"\b(She|He|It)\s+(do)\b",
                "does",
                "Subject-verb agreement: third-person singular requires 'does'.",
            ),
            (
                r"\b(She|He|It)\s+(have)\b",
                "has",
                "Subject-verb agreement: third-person singular requires 'has'.",
            ),
            (
                r"\b(She|He|It)\s+(don't)\b",
                "doesn't",
                "Subject-verb agreement: use 'doesn\\'t' with third-person singular.",
            ),
            (
                r"\b(I)\s+(has)\b",
                "have",
                "Subject-verb agreement: 'I' takes 'have', not 'has'.",
            ),
            (
                r"\b(I)\s+(is)\b",
                "am",
                "Subject-verb agreement: 'I' takes 'am', not 'is'.",
            ),
            (
                r"\b(They|We|You)\s+(is)\b",
                "are",
                "Subject-verb agreement: plural pronouns take 'are'.",
            ),
            (
                r"\b(They|We|You)\s+(was)\b",
                "were",
                "Subject-verb agreement: plural pronouns take 'were'.",
            ),
            (
                r"\b(yesterday|last\s+\w+)\s+(?:to\s+buying|buying)\b",
                "bought",
                "Past tense: use 'bought' for completed past actions.",
            ),
        ]

        for pattern, replacement, message in grammar_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                # If pattern captured 2 groups, target the second group (the verb)
                if match.lastindex and match.lastindex >= 2:
                    start = match.start(2)
                    length = len(match.group(2))
                    err_txt = match.group(2)
                else:
                    start = match.start()
                    length = len(match.group(0))
                    err_txt = match.group(0)

                issues.append(
                    GrammarIssue(
                        offset=start,
                        length=length,
                        error_text=err_txt,
                        message=message,
                        replacements=[replacement],
                        rule_id="SUBJECT_VERB_AGREEMENT",
                        category="Grammar",
                    )
                )

        # Sort issues strictly by offset ascending
        issues.sort(key=lambda x: (x.offset, x.length))
        return issues

    def _apply_reverse_corrections(self, text: str, issues: list[GrammarIssue]) -> str:
        """
        Applies suggestions from right-to-left (reverse offset).
        This guarantees string indices of preceding errors never shift or corrupt!
        """
        # ponytail: Sort reverse by offset so rightmost edits execute first
        sorted_issues = sorted(issues, key=lambda x: (x.offset, x.length), reverse=True)
        corrected = text
        last_replaced_start = len(text) + 1  # Boundary to prevent overlapping edits

        for issue in sorted_issues:
            if not issue.replacements or issue.offset < 0 or issue.length <= 0:
                continue
            start = issue.offset
            end = issue.offset + issue.length
            # Ensure replacement stays within text bounds and doesn't overlap a later replacement
            if end <= last_replaced_start and start >= 0 and end <= len(corrected):
                best_replacement = issue.replacements[0]
                corrected = corrected[:start] + best_replacement + corrected[end:]
                last_replaced_start = start

        return corrected

    def close(self) -> None:
        """Safely shuts down LanguageTool background server if active."""
        if self.tool is not None:
            try:
                # ponytail: LanguageTool server cleanup inside try/except prevents unhandled shutdown errors
                if hasattr(self.tool, "close"):
                    self.tool.close()
            except Exception as e:
                logger.warning("Exception during LanguageTool engine shutdown: %s", e)
            finally:
                self.tool = None
