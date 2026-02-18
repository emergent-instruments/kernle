"""Dump/export operations for Kernle."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SerializersMixin:
    """Dump/export operations for Kernle."""

    def dump(self, include_raw: bool = True, format: str = "markdown") -> str:
        """Export all memory to a readable format.

        Args:
            include_raw: Include raw entries in the dump
            format: Output format ("markdown" or "json")

        Returns:
            Formatted string of all memory
        """
        if format == "json":
            return self._dump_json(include_raw)
        else:
            return self._dump_markdown(include_raw)

    def _dump_markdown(self, include_raw: bool) -> str:
        """Export memory as markdown."""
        lines = []
        lines.append(f"# Memory Dump for {self.stack_id}")
        lines.append(f"_Exported at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
        lines.append("")

        # Values
        values = self._storage.get_values(limit=10000)
        if values:
            lines.append("## Values")
            for v in sorted(values, key=lambda x: x.priority, reverse=True):
                lines.append(f"- **{v.name}** (priority {v.priority}): {v.statement}")
            lines.append("")

        # Beliefs
        beliefs = self._storage.get_beliefs(limit=100)
        if beliefs:
            lines.append("## Beliefs")
            for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
                lines.append(f"- [{b.confidence:.0%}] {b.statement}")
            lines.append("")

        # Goals
        goals = self._storage.get_goals(status=None, limit=10000)
        if goals:
            lines.append("## Goals")
            for g in goals:
                status_icon = (
                    "✓" if g.status == "completed" else "○" if g.status == "active" else "⏸"
                )
                lines.append(f"- {status_icon} [{g.priority}] {g.title}")
                if g.description and g.description != g.title:
                    lines.append(f"  {g.description}")
            lines.append("")

        # Episodes
        episodes = self._storage.get_episodes(limit=10000)
        if episodes:
            lines.append("## Episodes")
            for e in episodes:
                date_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
                outcome_icon = (
                    "✓"
                    if e.outcome_type == "success"
                    else "✗" if e.outcome_type == "failure" else "○"
                )
                lines.append(f"### {outcome_icon} {e.objective}")
                lines.append(f"*{date_str}* | {e.outcome}")
                if e.lessons:
                    lines.append("**Lessons:**")
                    for lesson in e.lessons:
                        lines.append(f"  - {lesson}")
                if e.tags:
                    lines.append(f"Tags: {', '.join(e.tags)}")
                lines.append("")

        # Notes
        notes = self._storage.get_notes(limit=10000)
        if notes:
            lines.append("## Notes")
            for n in notes:
                date_str = n.created_at.strftime("%Y-%m-%d") if n.created_at else "unknown"
                lines.append(f"### [{n.note_type}] {date_str}")
                lines.append(n.content)
                if n.tags:
                    lines.append(f"Tags: {', '.join(n.tags)}")
                lines.append("")

        # Drives
        drives = self._storage.get_drives()
        if drives:
            lines.append("## Drives")
            for d in drives:
                bar = "█" * int(d.intensity * 10) + "░" * (10 - int(d.intensity * 10))
                focus = f" → {', '.join(d.focus_areas)}" if d.focus_areas else ""
                lines.append(f"- {d.drive_type}: [{bar}] {d.intensity:.0%}{focus}")
            lines.append("")

        # Relationships
        relationships = self._storage.get_relationships()
        if relationships:
            lines.append("## Relationships")
            for r in relationships:
                sentiment_str = f"{r.sentiment:+.2f}" if r.sentiment else "neutral"
                lines.append(f"- **{r.entity_name}** ({r.entity_type}): {sentiment_str}")
                if r.notes:
                    lines.append(f"  {r.notes}")
            lines.append("")

        # Raw entries
        if include_raw:
            raw_entries = self._storage.list_raw(limit=10000)
            if raw_entries:
                lines.append("## Raw Entries")
                for raw in raw_entries:
                    date_str = (
                        raw.timestamp.strftime("%Y-%m-%d %H:%M") if raw.timestamp else "unknown"
                    )
                    status = "✓" if raw.processed else "○"
                    lines.append(f"### {status} {date_str}")
                    lines.append(raw.content)
                    if raw.tags:
                        lines.append(f"Tags: {', '.join(raw.tags)}")
                    if raw.processed and raw.processed_into:
                        lines.append(f"Processed into: {', '.join(raw.processed_into)}")
                    lines.append("")

        return "\n".join(lines)

    def _dump_json(self, include_raw: bool) -> str:
        """Export memory as JSON with full meta-memory fields."""

        def _dt(dt: Optional[datetime]) -> Optional[str]:
            """Convert datetime to ISO string."""
            return dt.isoformat() if dt else None

        data = {
            "stack_id": self.stack_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "statement": v.statement,
                    "priority": v.priority,
                    "created_at": _dt(v.created_at),
                    "local_updated_at": _dt(v.local_updated_at),
                    "confidence": v.confidence,
                    "source_type": v.source_type,
                    "source_episodes": v.source_episodes,
                    "derived_from": v.derived_from,
                    "times_accessed": v.times_accessed,
                    "last_accessed": _dt(v.last_accessed),
                    "is_protected": v.is_protected,
                }
                for v in self._storage.get_values(limit=10000)
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "type": b.belief_type,
                    "confidence": b.confidence,
                    "created_at": _dt(b.created_at),
                    "local_updated_at": _dt(b.local_updated_at),
                    "source_type": b.source_type,
                    "source_episodes": b.source_episodes,
                    "derived_from": b.derived_from,
                    "times_accessed": b.times_accessed,
                    "last_accessed": _dt(b.last_accessed),
                    "is_protected": b.is_protected,
                    "supersedes": b.supersedes,
                    "superseded_by": b.superseded_by,
                    "times_reinforced": b.times_reinforced,
                    "is_active": b.is_active,
                }
                for b in self._storage.get_beliefs(limit=100)
            ],
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "description": g.description,
                    "priority": g.priority,
                    "status": g.status,
                    "created_at": _dt(g.created_at),
                    "local_updated_at": _dt(g.local_updated_at),
                    "confidence": g.confidence,
                    "source_type": g.source_type,
                    "source_episodes": g.source_episodes,
                    "derived_from": g.derived_from,
                    "times_accessed": g.times_accessed,
                    "last_accessed": _dt(g.last_accessed),
                    "is_protected": g.is_protected,
                }
                for g in self._storage.get_goals(status=None, limit=10000)
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "outcome_type": e.outcome_type,
                    "lessons": e.lessons,
                    "tags": e.tags,
                    "created_at": _dt(e.created_at),
                    "local_updated_at": _dt(e.local_updated_at),
                    "confidence": e.confidence,
                    "source_type": e.source_type,
                    "source_episodes": e.source_episodes,
                    "derived_from": e.derived_from,
                    "emotional_valence": e.emotional_valence,
                    "emotional_arousal": e.emotional_arousal,
                    "emotional_tags": e.emotional_tags,
                    "times_accessed": e.times_accessed,
                    "last_accessed": _dt(e.last_accessed),
                    "is_protected": e.is_protected,
                }
                for e in self._storage.get_episodes(limit=10000)
            ],
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "type": n.note_type,
                    "speaker": n.speaker,
                    "reason": n.reason,
                    "tags": n.tags,
                    "created_at": _dt(n.created_at),
                    "local_updated_at": _dt(n.local_updated_at),
                    "confidence": n.confidence,
                    "source_type": n.source_type,
                    "source_episodes": n.source_episodes,
                    "derived_from": n.derived_from,
                    "times_accessed": n.times_accessed,
                    "last_accessed": _dt(n.last_accessed),
                    "is_protected": n.is_protected,
                }
                for n in self._storage.get_notes(limit=10000)
            ],
            "drives": [
                {
                    "id": d.id,
                    "type": d.drive_type,
                    "intensity": d.intensity,
                    "focus_areas": d.focus_areas,
                    "created_at": _dt(d.created_at),
                    "updated_at": _dt(d.updated_at),
                    "local_updated_at": _dt(d.local_updated_at),
                    "confidence": d.confidence,
                    "source_type": d.source_type,
                    "derived_from": d.derived_from,
                    "times_accessed": d.times_accessed,
                    "last_accessed": _dt(d.last_accessed),
                    "is_protected": d.is_protected,
                }
                for d in self._storage.get_drives()
            ],
            "relationships": [
                {
                    "id": r.id,
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "relationship_type": r.relationship_type,
                    "sentiment": r.sentiment,
                    "notes": r.notes,
                    "interaction_count": r.interaction_count,
                    "last_interaction": _dt(r.last_interaction),
                    "created_at": _dt(r.created_at),
                    "local_updated_at": _dt(r.local_updated_at),
                    "confidence": r.confidence,
                    "source_type": r.source_type,
                    "derived_from": r.derived_from,
                    "times_accessed": r.times_accessed,
                    "last_accessed": _dt(r.last_accessed),
                    "is_protected": r.is_protected,
                }
                for r in self._storage.get_relationships()
            ],
        }

        if include_raw:
            data["raw_entries"] = [
                {
                    "id": r.id,
                    "content": r.content,
                    "timestamp": _dt(r.timestamp),
                    "source": r.source,
                    "processed": r.processed,
                    "processed_into": r.processed_into,
                    "tags": r.tags,
                    "local_updated_at": _dt(r.local_updated_at),
                    "confidence": r.confidence,
                    "source_type": r.source_type,
                }
                for r in self._storage.list_raw(limit=10000)
            ]

        return json.dumps(data, indent=2, default=str)

    def export(self, path: str, include_raw: bool = True, format: str = "markdown"):
        """Export memory to a file.

        Args:
            path: Path to export file
            include_raw: Include raw entries
            format: Output format ("markdown" or "json")
        """
        content = self.dump(include_raw=include_raw, format=format)

        # Determine format from extension if not specified
        if format == "markdown" and path.endswith(".json"):
            format = "json"
            content = self.dump(include_raw=include_raw, format="json")
        elif format == "json" and (path.endswith(".md") or path.endswith(".markdown")):
            format = "markdown"
            content = self.dump(include_raw=include_raw, format="markdown")

        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(content, encoding="utf-8")

    def export_full(
        self,
        path: Optional[str] = None,
        format: str = "markdown",
        include_raw: bool = False,
    ) -> str:
        """Export complete agent context to a single file.

        Unlike export() which dumps memory layers, and export_cache() which
        produces a curated bootstrap cache, export_full() assembles ALL
        memory layers including boot config, self-narratives, trust
        assessments, playbooks, and checkpoint into one comprehensive file.

        Args:
            path: If provided, write to this file. Otherwise return string.
            format: Output format ("markdown" or "json")
            include_raw: Include raw entries (default: False)

        Returns:
            The exported content string
        """
        # Auto-detect format from file extension if path is provided
        if path:
            if path.endswith(".json"):
                format = "json"
            elif path.endswith(".md") or path.endswith(".markdown"):
                format = "markdown"

        if format == "json":
            content = self._export_full_json(include_raw)
        else:
            content = self._export_full_markdown(include_raw)

        if path:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(content, encoding="utf-8")

        return content

    def _export_full_markdown(self, include_raw: bool) -> str:
        """Export complete agent context as markdown."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Full Agent Context — {self.stack_id}",
            "",
            f"_Exported at {now_str}_",
            "",
        ]

        # Boot config
        boot_lines = self._format_boot_section()
        if boot_lines:
            lines.extend(boot_lines)

        # Self-narratives (active)
        narratives = self._storage.list_self_narratives(self.stack_id, active_only=True)
        if narratives:
            lines.append("## Self-Narratives")
            for n in narratives:
                lines.append(f"### {n.narrative_type.title()}")
                lines.append(n.content)
                if n.key_themes:
                    lines.append(f"Themes: {', '.join(n.key_themes)}")
                if n.unresolved_tensions:
                    lines.append(f"Tensions: {', '.join(n.unresolved_tensions)}")
                lines.append("")

        # Values
        values = self._storage.get_values(limit=10000)
        if values:
            lines.append("## Values")
            for v in sorted(
                values, key=lambda x: x.priority if x.priority is not None else 0, reverse=True
            ):
                lines.append(f"- **{v.name}** (priority {v.priority or 0}): {v.statement}")
            lines.append("")

        # Beliefs
        beliefs = self._storage.get_beliefs(limit=10000)
        if beliefs:
            lines.append("## Beliefs")
            for b in sorted(
                beliefs,
                key=lambda x: x.confidence if x.confidence is not None else 0.0,
                reverse=True,
            ):
                lines.append(f"- [{b.confidence:.0%}] {b.statement}")
            lines.append("")

        # Goals
        goals = self._storage.get_goals(status=None, limit=10000)
        if goals:
            lines.append("## Goals")
            for g in goals:
                status_icon = (
                    "+" if g.status == "completed" else "o" if g.status == "active" else "-"
                )
                lines.append(f"- {status_icon} [{g.priority}] {g.title}")
                if g.description and g.description != g.title:
                    lines.append(f"  {g.description}")
            lines.append("")

        # Drives
        drives = self._storage.get_drives()
        if drives:
            lines.append("## Drives")
            for d in drives:
                focus = f" -> {', '.join(d.focus_areas)}" if d.focus_areas else ""
                lines.append(f"- {d.drive_type}: {d.intensity:.0%}{focus}")
            lines.append("")

        # Relationships
        relationships = self._storage.get_relationships()
        if relationships:
            lines.append("## Relationships")
            for r in relationships:
                sentiment_str = f"{r.sentiment:+.2f}" if r.sentiment else "neutral"
                lines.append(f"- **{r.entity_name}** ({r.entity_type}): {sentiment_str}")
                if r.notes:
                    lines.append(f"  {r.notes}")
            lines.append("")

        # Episodes
        episodes = self._storage.get_episodes(limit=10000)
        if episodes:
            lines.append("## Episodes")
            for e in episodes:
                date_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
                outcome_icon = (
                    "+"
                    if e.outcome_type == "success"
                    else "x" if e.outcome_type == "failure" else "o"
                )
                lines.append(f"### {outcome_icon} {e.objective}")
                lines.append(f"*{date_str}* | {e.outcome}")
                if e.lessons:
                    lines.append("**Lessons:**")
                    for lesson in e.lessons:
                        lines.append(f"  - {lesson}")
                if e.tags:
                    lines.append(f"Tags: {', '.join(e.tags)}")
                lines.append("")

        # Notes
        notes = self._storage.get_notes(limit=10000)
        if notes:
            lines.append("## Notes")
            for n in notes:
                date_str = n.created_at.strftime("%Y-%m-%d") if n.created_at else "unknown"
                lines.append(f"### [{n.note_type}] {date_str}")
                lines.append(n.content)
                if n.tags:
                    lines.append(f"Tags: {', '.join(n.tags)}")
                lines.append("")

        # Suggestions (all statuses)
        suggestions = self._storage.get_suggestions(limit=10000)
        if suggestions:
            lines.append("## Suggestions")
            pending = [s for s in suggestions if s.status == "pending"]
            resolved = [s for s in suggestions if s.status != "pending"]
            if pending:
                lines.append(f"### Pending ({len(pending)})")
                for s in pending:
                    preview = ""
                    if s.memory_type == "episode":
                        preview = s.content.get("objective", "")[:60]
                    elif s.memory_type == "belief":
                        preview = s.content.get("statement", "")[:60]
                    else:
                        preview = s.content.get("content", "")[:60]
                    lines.append(f"- [{s.confidence:.0%}] {s.memory_type}: {preview}")
                lines.append("")
            if resolved:
                lines.append(f"### Resolved ({len(resolved)})")
                for s in resolved:
                    preview = ""
                    if s.memory_type == "episode":
                        preview = s.content.get("objective", "")[:60]
                    elif s.memory_type == "belief":
                        preview = s.content.get("statement", "")[:60]
                    else:
                        preview = s.content.get("content", "")[:60]
                    promoted = f" -> {s.promoted_to}" if s.promoted_to else ""
                    reason = f" ({s.resolution_reason})" if s.resolution_reason else ""
                    lines.append(f"- [{s.status}] {s.memory_type}: {preview}{promoted}{reason}")
                lines.append("")

        # Trust assessments
        assessments = self._storage.get_trust_assessments()
        if assessments:
            lines.append("## Trust Assessments")
            for a in assessments:
                dims = ", ".join(
                    f"{d}: {info.get('score', '?')}" if isinstance(info, dict) else f"{d}: {info}"
                    for d, info in a.dimensions.items()
                )
                lines.append(f"- **{a.entity}**: {dims}")
                if a.authority:
                    lines.append(f"  Authority: {a.authority}")
            lines.append("")

        # Playbooks
        playbooks = self._storage.list_playbooks(limit=10000)
        if playbooks:
            lines.append("## Playbooks")
            for p in playbooks:
                lines.append(f"### {p.name}")
                lines.append(f"_{p.description}_")
                lines.append(
                    f"Mastery: {p.mastery_level} | Used: {p.times_used} | Success: {p.success_rate:.0%}"
                )
                if p.trigger_conditions:
                    lines.append("**Triggers:**")
                    for t in p.trigger_conditions:
                        lines.append(f"  - {t}")
                if p.steps:
                    lines.append("**Steps:**")
                    for i, step in enumerate(p.steps, 1):
                        action = (
                            step.get("action", str(step)) if isinstance(step, dict) else str(step)
                        )
                        lines.append(f"  {i}. {action}")
                if p.failure_modes:
                    lines.append("**Failure modes:**")
                    for fm in p.failure_modes:
                        lines.append(f"  - {fm}")
                lines.append("")

        # Checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint:
            lines.append("## Checkpoint")
            lines.append(f"**Task**: {checkpoint.get('current_task', 'unknown')}")
            if checkpoint.get("context"):
                lines.append(f"**Context**: {checkpoint['context']}")
            if checkpoint.get("pending"):
                lines.append("**Pending**:")
                for p in checkpoint["pending"]:
                    lines.append(f"  - {p}")
            lines.append("")

        # Raw entries (optional)
        if include_raw:
            raw_entries = self._storage.list_raw(limit=10000)
            if raw_entries:
                lines.append("## Raw Entries")
                for raw in raw_entries:
                    ts = raw.captured_at or raw.timestamp
                    date_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "unknown"
                    status = "+" if raw.processed else "o"
                    lines.append(f"### {status} {date_str}")
                    lines.append(raw.blob or raw.content or "")
                    if raw.processed and raw.processed_into:
                        lines.append(f"Processed into: {', '.join(raw.processed_into)}")
                    lines.append("")

        return "\n".join(lines)

    def _export_full_json(self, include_raw: bool) -> str:
        """Export complete agent context as JSON with full metadata."""

        def _dt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        # Self-narratives
        narratives = self._storage.list_self_narratives(self.stack_id, active_only=False)

        # Trust assessments
        assessments = self._storage.get_trust_assessments()

        # Playbooks
        playbooks = self._storage.list_playbooks(limit=10000)

        data = {
            "stack_id": self.stack_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format": "export-full",
            "boot_config": self.boot_list(),
            "self_narratives": [
                {
                    "id": n.id,
                    "content": n.content,
                    "narrative_type": n.narrative_type,
                    "epoch_id": n.epoch_id,
                    "key_themes": n.key_themes,
                    "unresolved_tensions": n.unresolved_tensions,
                    "is_active": n.is_active,
                    "supersedes": n.supersedes,
                    "created_at": _dt(n.created_at),
                }
                for n in narratives
            ],
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "statement": v.statement,
                    "priority": v.priority,
                    "created_at": _dt(v.created_at),
                    "local_updated_at": _dt(v.local_updated_at),
                    "confidence": v.confidence,
                    "source_type": v.source_type,
                    "source_episodes": v.source_episodes,
                    "derived_from": v.derived_from,
                    "times_accessed": v.times_accessed,
                    "last_accessed": _dt(v.last_accessed),
                    "is_protected": v.is_protected,
                }
                for v in self._storage.get_values(limit=10000)
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "type": b.belief_type,
                    "confidence": b.confidence,
                    "created_at": _dt(b.created_at),
                    "local_updated_at": _dt(b.local_updated_at),
                    "source_type": b.source_type,
                    "source_episodes": b.source_episodes,
                    "derived_from": b.derived_from,
                    "times_accessed": b.times_accessed,
                    "last_accessed": _dt(b.last_accessed),
                    "is_protected": b.is_protected,
                    "supersedes": b.supersedes,
                    "superseded_by": b.superseded_by,
                    "times_reinforced": b.times_reinforced,
                    "is_active": b.is_active,
                }
                for b in self._storage.get_beliefs(limit=10000)
            ],
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "description": g.description,
                    "priority": g.priority,
                    "status": g.status,
                    "created_at": _dt(g.created_at),
                    "local_updated_at": _dt(g.local_updated_at),
                    "confidence": g.confidence,
                    "source_type": g.source_type,
                    "source_episodes": g.source_episodes,
                    "derived_from": g.derived_from,
                    "times_accessed": g.times_accessed,
                    "last_accessed": _dt(g.last_accessed),
                    "is_protected": g.is_protected,
                }
                for g in self._storage.get_goals(status=None, limit=10000)
            ],
            "drives": [
                {
                    "id": d.id,
                    "type": d.drive_type,
                    "intensity": d.intensity,
                    "focus_areas": d.focus_areas,
                    "created_at": _dt(d.created_at),
                    "updated_at": _dt(d.updated_at),
                    "local_updated_at": _dt(d.local_updated_at),
                    "confidence": d.confidence,
                    "source_type": d.source_type,
                    "derived_from": d.derived_from,
                    "times_accessed": d.times_accessed,
                    "last_accessed": _dt(d.last_accessed),
                    "is_protected": d.is_protected,
                }
                for d in self._storage.get_drives()
            ],
            "relationships": [
                {
                    "id": r.id,
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "relationship_type": r.relationship_type,
                    "sentiment": r.sentiment,
                    "notes": r.notes,
                    "interaction_count": r.interaction_count,
                    "last_interaction": _dt(r.last_interaction),
                    "created_at": _dt(r.created_at),
                    "local_updated_at": _dt(r.local_updated_at),
                    "confidence": r.confidence,
                    "source_type": r.source_type,
                    "derived_from": r.derived_from,
                    "times_accessed": r.times_accessed,
                    "last_accessed": _dt(r.last_accessed),
                    "is_protected": r.is_protected,
                }
                for r in self._storage.get_relationships()
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "outcome_type": e.outcome_type,
                    "lessons": e.lessons,
                    "tags": e.tags,
                    "created_at": _dt(e.created_at),
                    "local_updated_at": _dt(e.local_updated_at),
                    "confidence": e.confidence,
                    "source_type": e.source_type,
                    "source_episodes": e.source_episodes,
                    "derived_from": e.derived_from,
                    "emotional_valence": e.emotional_valence,
                    "emotional_arousal": e.emotional_arousal,
                    "emotional_tags": e.emotional_tags,
                    "times_accessed": e.times_accessed,
                    "last_accessed": _dt(e.last_accessed),
                    "is_protected": e.is_protected,
                }
                for e in self._storage.get_episodes(limit=10000)
            ],
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "type": n.note_type,
                    "speaker": n.speaker,
                    "reason": n.reason,
                    "tags": n.tags,
                    "created_at": _dt(n.created_at),
                    "local_updated_at": _dt(n.local_updated_at),
                    "confidence": n.confidence,
                    "source_type": n.source_type,
                    "source_episodes": n.source_episodes,
                    "derived_from": n.derived_from,
                    "times_accessed": n.times_accessed,
                    "last_accessed": _dt(n.last_accessed),
                    "is_protected": n.is_protected,
                }
                for n in self._storage.get_notes(limit=10000)
            ],
            "suggestions": [
                {
                    "id": s.id,
                    "memory_type": s.memory_type,
                    "content": s.content,
                    "confidence": s.confidence,
                    "source_raw_ids": s.source_raw_ids,
                    "status": s.status,
                    "created_at": _dt(s.created_at),
                    "resolved_at": _dt(s.resolved_at),
                    "resolution_reason": s.resolution_reason,
                    "promoted_to": s.promoted_to,
                }
                for s in self._storage.get_suggestions(limit=10000)
            ],
            "trust_assessments": [
                {
                    "id": a.id,
                    "entity": a.entity,
                    "dimensions": a.dimensions,
                    "authority": a.authority or [],
                    "evidence_episode_ids": a.evidence_episode_ids or [],
                    "last_updated": _dt(a.last_updated),
                    "created_at": _dt(a.created_at),
                }
                for a in assessments
            ],
            "playbooks": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "trigger_conditions": p.trigger_conditions,
                    "steps": p.steps,
                    "failure_modes": p.failure_modes,
                    "recovery_steps": p.recovery_steps,
                    "mastery_level": p.mastery_level,
                    "times_used": p.times_used,
                    "success_rate": p.success_rate,
                    "source_episodes": p.source_episodes,
                    "tags": p.tags,
                    "confidence": p.confidence,
                    "last_used": _dt(p.last_used),
                    "created_at": _dt(p.created_at),
                }
                for p in playbooks
            ],
            "checkpoint": self.load_checkpoint(),
        }

        if include_raw:
            data["raw_entries"] = [
                {
                    "id": r.id,
                    "blob": r.blob or r.content,
                    "captured_at": _dt(r.captured_at or r.timestamp),
                    "source": r.source,
                    "processed": r.processed,
                    "processed_into": r.processed_into,
                    "local_updated_at": _dt(r.local_updated_at),
                }
                for r in self._storage.list_raw(limit=10000)
            ]

        return json.dumps(data, indent=2, default=str)

    def export_cache(
        self,
        path: Optional[str] = None,
        min_confidence: float = 0.4,
        max_beliefs: int = 50,
        include_checkpoint: bool = True,
    ) -> str:
        """Export a curated MEMORY.md cache from beliefs, values, and goals.

        This produces a read-only bootstrap cache for workspace injection.
        The output is designed to give an agent immediate context before
        `kernle load` runs. It is NOT a full memory dump — just the
        high-signal layers.

        Args:
            path: If provided, write to this file. Otherwise return string.
            min_confidence: Minimum belief confidence to include (default: 0.4)
            max_beliefs: Maximum number of beliefs to include (default: 50)
            include_checkpoint: Include last checkpoint if available

        Returns:
            The markdown content (also written to path if provided)
        """
        # Validate inputs
        min_confidence = max(0.0, min(1.0, min_confidence))
        max_beliefs = max(1, min(1000, max_beliefs))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# MEMORY.md — Long-Term Memory",
            "",
            f"<!-- AUTO-GENERATED by `kernle export-cache` at {now_str} -->",
            "<!-- Do not edit manually. Source of truth is Kernle. -->",
            f"<!-- Regenerate with: kernle -s {self.stack_id} export-cache -->",
            "",
        ]

        # Boot config (always first — this is the pre-load config)
        boot_lines = self._format_boot_section()
        if boot_lines:
            lines.extend(boot_lines)

        # Values (highest priority first)
        values = self._storage.get_values(limit=20)
        if values:
            lines.append("## Values")
            for v in sorted(
                values, key=lambda x: x.priority if x.priority is not None else 0, reverse=True
            ):
                safe_stmt = v.statement.replace("\n", " ").replace("\r", "") if v.statement else ""
                lines.append(f"- **{v.name}** (priority {v.priority or 0}): {safe_stmt}")
            lines.append("")

        # Goals (active only)
        goals = self._storage.get_goals(status="active", limit=20)
        if goals:
            lines.append("## Goals")
            for g in sorted(
                goals, key=lambda x: x.priority if x.priority is not None else 0, reverse=True
            ):
                desc = f" — {g.description}" if g.description and g.description != g.title else ""
                lines.append(f"- [{g.priority}] {g.title}{desc}")
            lines.append("")

        # Beliefs (filtered by confidence, sorted desc)
        # Fetch all beliefs since storage orders by created_at, not confidence
        beliefs = self._storage.get_beliefs(limit=max(max_beliefs * 3, 200))
        if beliefs:
            filtered = [b for b in beliefs if (b.confidence or 0) >= min_confidence]
            filtered.sort(
                key=lambda x: x.confidence if x.confidence is not None else 0.0, reverse=True
            )
            filtered = filtered[:max_beliefs]

            if filtered:
                lines.append("## Beliefs")
                for b in filtered:
                    # Strip newlines to prevent markdown structure injection
                    safe_statement = b.statement.replace("\n", " ").replace("\r", "")
                    lines.append(f"- [{b.confidence:.0%}] {safe_statement}")
                lines.append("")

        # Relationships (top by interaction count)
        relationships = self._storage.get_relationships()
        if relationships:
            # Sort by interaction count descending
            sorted_rels = sorted(
                relationships,
                key=lambda r: r.interaction_count,
                reverse=True,
            )[:10]
            if sorted_rels:
                lines.append("## Key Relationships")
                for r in sorted_rels:
                    notes_str = (
                        f" — {r.notes[:80]}..."
                        if r.notes and len(r.notes) > 80
                        else (f" — {r.notes}" if r.notes else "")
                    )
                    lines.append(f"- **{r.entity_name}** ({r.entity_type}){notes_str}")
                lines.append("")

        # Checkpoint (for session continuity)
        if include_checkpoint:
            cp = self.load_checkpoint()
            if cp:
                lines.append("## Last Checkpoint")
                lines.append(f"**Task**: {cp.get('current_task', 'unknown')}")
                if cp.get("context"):
                    lines.append(f"**Context**: {cp['context']}")
                if cp.get("pending"):
                    lines.append("**Pending**:")
                    for p in cp["pending"]:
                        lines.append(f"  - {p}")
                lines.append("")

        content = "\n".join(lines)

        if path:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(content, encoding="utf-8")

        return content
