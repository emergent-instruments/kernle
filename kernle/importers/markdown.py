"""Markdown importer for Kernle.

Parses markdown files with sections like:
- ## Beliefs
- ## Episodes / ## Lessons
- ## Notes / ## Decisions
- ## Values / ## Principles
- ## Goals / ## Tasks
- ## Raw / ## Thoughts
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.core.writers import IMPORT_TOKEN

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kernle import Kernle


@dataclass
class ImportItem:
    """A parsed item ready for import."""

    type: str  # episode, note, belief, value, goal, raw
    content: str = ""
    # Type-specific fields
    objective: str = ""
    outcome: str = ""
    lesson: Optional[str] = None
    statement: str = ""
    confidence: float = 0.7
    name: str = ""
    description: str = ""
    note_type: str = "note"
    status: str = "active"
    priority: int = 50
    source: str = "import"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownImporter:
    """Import memories from markdown files.

    Supports the markdown format exported by `kernle export` as well as
    common MEMORY.md patterns used in AI projects.
    """

    def __init__(self, file_path: str):
        """Initialize with path to markdown file.

        Args:
            file_path: Path to the markdown file to import
        """
        self.file_path = Path(file_path).expanduser()
        self.items: List[ImportItem] = []

    def parse(self) -> List[ImportItem]:
        """Parse the markdown file and return importable items.

        Returns:
            List of ImportItem objects

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        self.items = parse_markdown(content, origin_file=str(self.file_path))
        return self.items

    def import_to(self, k: "Kernle", dry_run: bool = False) -> Dict[str, int]:
        """Import parsed items into a Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import, just return counts

        Returns:
            Dict with counts of items imported by type
        """
        if not self.items:
            self.parse()

        counts: Dict[str, int] = {}
        errors: List[str] = []

        for item in self.items:
            try:
                if not dry_run:
                    _import_item(item, k)
                counts[item.type] = counts.get(item.type, 0) + 1
            except Exception as e:
                logger.debug("Markdown import item %s failed: %s", item.type, e, exc_info=True)
                errors.append(f"{item.type}: {str(e)[:50]}")

        return counts


def parse_markdown(
    content: str,
    origin_file: Optional[str] = None,
) -> List[ImportItem]:
    """Parse markdown content into importable items.

    Detects sections like:
    - ## Episodes, ## Lessons -> episode
    - ## Decisions, ## Notes, ## Insights -> note
    - ## Beliefs -> belief
    - ## Values, ## Principles -> value
    - ## Goals, ## Tasks -> goal
    - ## Raw, ## Thoughts, ## Scratch -> raw
    - Unstructured text -> raw

    Args:
        content: Markdown content to parse
        origin_file: Optional source file path for provenance tracking

    Returns:
        List of ImportItem objects
    """
    items: List[ImportItem] = []

    # Split into sections by ## headers
    sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

    # First section (before any ##) is preamble
    if sections[0].strip():
        preamble = sections[0].strip()
        for para in _split_paragraphs(preamble):
            if para.strip():
                item = ImportItem(type="raw", content=para.strip(), source="preamble")
                if origin_file:
                    item.metadata["_import_meta"] = {
                        "origin_file": origin_file,
                        "origin_section": "preamble",
                    }
                items.append(item)

    # Process header sections
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        header = sections[i].strip()
        header_lower = header.lower()
        section_content = sections[i + 1].strip()

        if not section_content:
            continue

        # Determine type from header
        if any(h in header_lower for h in ["episode", "lesson", "experience", "event"]):
            section_items = _parse_episodes(section_content)
        elif any(h in header_lower for h in ["decision", "note", "insight", "observation"]):
            section_items = _parse_notes(section_content, header_lower)
        elif "belief" in header_lower:
            section_items = _parse_beliefs(section_content)
        elif any(h in header_lower for h in ["value", "principle"]):
            section_items = _parse_values(section_content)
        elif any(h in header_lower for h in ["goal", "objective", "todo", "task"]):
            section_items = _parse_goals(section_content)
        elif any(h in header_lower for h in ["raw", "thought", "scratch", "draft", "idea"]):
            section_items = _parse_raw(section_content)
        else:
            # Unknown section - treat as raw
            section_items = _parse_raw(section_content)

        # Annotate section items with origin metadata
        if origin_file:
            for item in section_items:
                item.metadata["_import_meta"] = {
                    "origin_file": origin_file,
                    "origin_section": header,
                }

        items.extend(section_items)

    return items


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _parse_episodes(content: str) -> List[ImportItem]:
    """Parse episode entries from section content."""
    items = []

    # Look for bullet points or numbered items
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract lesson (after -> or "Lesson:")
        lesson = None
        if "->" in entry:
            parts = entry.split("->", 1)
            entry = parts[0].strip()
            lesson = parts[1].strip()
        elif "lesson:" in entry.lower():
            # Match (Lesson: X) or just Lesson: X
            match = re.search(r"\(lesson:\s*([^)]+)\)", entry, re.IGNORECASE)
            if match:
                lesson = match.group(1).strip()
                entry = re.sub(r"\(lesson:\s*[^)]+\)", "", entry, flags=re.IGNORECASE).strip()
            else:
                # No parentheses version
                match = re.search(r"lesson:\s*(.+)", entry, re.IGNORECASE)
                if match:
                    lesson = match.group(1).strip()
                    entry = re.sub(r"lesson:\s*.+", "", entry, flags=re.IGNORECASE).strip()

        # Check for outcome type markers
        outcome_type = None
        if "[success]" in entry.lower():
            outcome_type = "success"
            entry = re.sub(r"\[success\]", "", entry, flags=re.IGNORECASE).strip()
        elif "[failure]" in entry.lower() or "[failed]" in entry.lower():
            outcome_type = "failure"
            entry = re.sub(r"\[failure\]|\[failed\]", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            ImportItem(
                type="episode",
                objective=entry[:200] if len(entry) > 200 else entry,
                outcome=entry,
                lesson=lesson,
                source="episodes section",
                metadata={"outcome_type": outcome_type} if outcome_type else {},
            )
        )

    return items


def _parse_notes(content: str, header: str) -> List[ImportItem]:
    """Parse note entries from section content."""
    items = []

    # Determine note type from header
    if "decision" in header:
        note_type = "decision"
    elif "insight" in header:
        note_type = "insight"
    elif "observation" in header:
        note_type = "observation"
    else:
        note_type = "note"

    # Split by bullets or paragraphs
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append(
            ImportItem(type="note", content=entry, note_type=note_type, source=f"{header} section")
        )

    return items


def _parse_beliefs(content: str) -> List[ImportItem]:
    """Parse belief entries from section content.

    Supports confidence formats:
    - (80%) or [0.8] after the statement
    - (confidence: 0.9) format
    """
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract confidence
        confidence = 0.7  # default

        # Format: (80%) or [80%]
        conf_match = re.search(r"\((\d+)%\)|\[(\d+)%\]", entry)
        if conf_match:
            pct = conf_match.group(1) or conf_match.group(2)
            confidence = int(pct) / 100
            entry = re.sub(r"\(\d+%\)|\[\d+%\]", "", entry).strip()
        else:
            # Format: [0.8] or (0.8)
            conf_match = re.search(r"\[(\d*\.?\d+)\]|\((\d*\.?\d+)\)", entry)
            if conf_match:
                val = conf_match.group(1) or conf_match.group(2)
                confidence = float(val)
                if confidence > 1:
                    confidence = confidence / 100
                entry = re.sub(r"\[\d*\.?\d+\]|\(\d*\.?\d+\)", "", entry).strip()
            else:
                # Format: (confidence: 0.9)
                conf_match = re.search(r"\(confidence:\s*(\d*\.?\d+)\)", entry, re.IGNORECASE)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    if confidence > 1:
                        confidence = confidence / 100
                    entry = re.sub(
                        r"\(confidence:\s*\d*\.?\d+\)", "", entry, flags=re.IGNORECASE
                    ).strip()

        # Remove "I believe" prefix if present
        entry = re.sub(r"^I believe\s+", "", entry, flags=re.IGNORECASE)

        items.append(
            ImportItem(
                type="belief",
                statement=entry,
                confidence=min(1.0, max(0.0, confidence)),  # Clamp to [0, 1]
                source="beliefs section",
            )
        )

    return items


def _parse_values(content: str) -> List[ImportItem]:
    """Parse value entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for name: description format
        if ":" in entry:
            name, desc = entry.split(":", 1)
            name = name.strip()
            desc = desc.strip()
        else:
            name = entry[:50]
            desc = entry

        # Check for priority (priority: N) or [priority: N]
        priority = 50
        priority_match = re.search(r"\(?priority[:\s]+(\d+)\)?", entry, re.IGNORECASE)
        if priority_match:
            priority = int(priority_match.group(1))
            entry = re.sub(r"\(?priority[:\s]+\d+\)?", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            ImportItem(
                type="value",
                name=name,
                description=desc,
                priority=priority,
                source="values section",
            )
        )

    return items


def _parse_goals(content: str) -> List[ImportItem]:
    """Parse goal entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for [done] or [x] markers
        status = "active"
        if re.search(r"\[x\]|\[done\]|\[complete\]", entry, re.IGNORECASE):
            status = "completed"
            entry = re.sub(r"\[x\]|\[done\]|\[complete\]", "", entry, flags=re.IGNORECASE).strip()
        elif re.search(r"\[paused\]|\[hold\]", entry, re.IGNORECASE):
            status = "paused"
            entry = re.sub(r"\[paused\]|\[hold\]", "", entry, flags=re.IGNORECASE).strip()

        # Check for priority
        priority = "medium"
        if re.search(r"\[high\]|\[urgent\]|\[p1\]", entry, re.IGNORECASE):
            priority = "high"
            entry = re.sub(r"\[high\]|\[urgent\]|\[p1\]", "", entry, flags=re.IGNORECASE).strip()
        elif re.search(r"\[low\]|\[p3\]", entry, re.IGNORECASE):
            priority = "low"
            entry = re.sub(r"\[low\]|\[p3\]", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            ImportItem(
                type="goal",
                description=entry,
                status=status,
                source="goals section",
                metadata={"priority": priority},
            )
        )

    return items


def _parse_raw(content: str) -> List[ImportItem]:
    """Parse raw entries from section content."""
    items = []

    # Check for bullet points first
    if re.search(r"^[-*]\s+", content, flags=re.MULTILINE):
        entries = re.split(r"^[-*]\s+", content, flags=re.MULTILINE)
    else:
        entries = _split_paragraphs(content)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append(ImportItem(type="raw", content=entry, source="raw section"))

    return items


def _import_item(item: ImportItem, k: "Kernle") -> Optional[str]:
    """Import a single item into Kernle.

    Args:
        item: The ImportItem to import
        k: Kernle instance

    Returns:
        The ID of the created memory, or None
    """
    t = item.type

    if t == "episode":
        lessons = [item.lesson] if item.lesson else None
        # Fold outcome_type into tags instead of passing as kwarg
        tags = None
        outcome_type = item.metadata.get("outcome_type")
        if outcome_type:
            tags = [f"outcome:{outcome_type}"]
        return k.episode(
            objective=item.objective,
            outcome=item.outcome or item.objective,
            lessons=lessons,
            tags=tags,
        )
    elif t == "note":
        return k.note(content=item.content, type=item.note_type)
    elif t == "belief":
        return k.belief(
            statement=item.statement,
            confidence=item.confidence,
            source_type="imported",
            _import_token=IMPORT_TOKEN,
        )
    elif t == "value":
        return k.value(
            name=item.name,
            statement=item.description,
            priority=item.priority,
            source_type="imported",
            _import_token=IMPORT_TOKEN,
        )
    elif t == "goal":
        return k.goal(
            title=item.description,
            description=item.description,
            priority=item.metadata.get("priority", "medium"),
            source_type="imported",
            _import_token=IMPORT_TOKEN,
        )
    elif t == "raw":
        return k.raw(item.content)

    return None
