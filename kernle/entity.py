"""
Entity — the Core coordinator/bus for kernle.

Entity implements CoreProtocol. It manages stack composition,
plugin lifecycle, model binding, and routes memory operations
to the active stack with provenance enforcement.

The entity is not the agent. The entity is the composition —
no single component IS the entity.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from kernle.core.enrichment import (
    build_derived_from,
    clamp_confidence,
    clamp_intensity,
    format_note_content,
    infer_outcome_type,
    normalize_belief_type,
    normalize_note_type,
    normalize_source_type,
    validate_drive_type,
    validate_goal_type,
)
from kernle.discovery import discover_plugins
from kernle.logging_config import log_save
from kernle.protocols import (
    Binding,
    InferenceService,
    ModelProtocol,
    NoActiveStackError,
    PluginHealth,
    PluginInfo,
    PluginProtocol,
    SearchResult,
    StackInfo,
    StackProtocol,
    SyncResult,
    ToolDefinition,
)
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    SourceType,
    TrustAssessment,
    Value,
)
from kernle.utils import get_kernle_home

_normalize_source_type = normalize_source_type

logger = logging.getLogger(__name__)


_ID_LOCK = threading.Lock()
_ID_SEQUENCE = 0
_ID_LAST_TIMESTAMP_MS = 0
_ID_NAMESPACE_UUID = uuid.UUID("2f6dce7e-c8f5-4f6e-9f2e-9b6f3d8c0c3a")
_ID_NAMESPACE = _ID_NAMESPACE_UUID.hex[:12]
_ID_SEQUENCE_MAX = 1_000_000


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ID_SEQUENCE_WARN_THRESHOLD = int(_ID_SEQUENCE_MAX * 0.9)  # 900,000
_ID_CLOCK_WAS_PINNED = False


def _generate_id() -> str:
    global _ID_SEQUENCE, _ID_LAST_TIMESTAMP_MS, _ID_CLOCK_WAS_PINNED
    with _ID_LOCK:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if now_ms < _ID_LAST_TIMESTAMP_MS:
            # Backward clock jump detected -- pin to last known timestamp
            # to preserve monotonicity.
            delta = _ID_LAST_TIMESTAMP_MS - now_ms
            logger.warning("Clock jump detected: %d ms backward", delta)
            now_ms = _ID_LAST_TIMESTAMP_MS
            _ID_CLOCK_WAS_PINNED = True

        if now_ms == _ID_LAST_TIMESTAMP_MS:
            _ID_SEQUENCE += 1
            if _ID_SEQUENCE >= _ID_SEQUENCE_MAX:
                raise RuntimeError(
                    f"ID sequence exhausted: {_ID_SEQUENCE_MAX} IDs generated "
                    f"within millisecond {now_ms}. Cannot guarantee uniqueness."
                )
            if _ID_SEQUENCE >= _ID_SEQUENCE_WARN_THRESHOLD:
                logger.warning(
                    "ID sequence nearing limit: %d / %d (%.0f%%)",
                    _ID_SEQUENCE,
                    _ID_SEQUENCE_MAX,
                    _ID_SEQUENCE / _ID_SEQUENCE_MAX * 100,
                )
        else:
            if _ID_CLOCK_WAS_PINNED:
                logger.info(
                    "Clock recovered: advanced to %d (was pinned at %d)",
                    now_ms,
                    _ID_LAST_TIMESTAMP_MS,
                )
                _ID_CLOCK_WAS_PINNED = False
            _ID_LAST_TIMESTAMP_MS = now_ms
            _ID_SEQUENCE = 0

        digest = uuid.uuid5(
            _ID_NAMESPACE_UUID,
            f"{_ID_NAMESPACE}-{now_ms:013d}-{_ID_SEQUENCE:06d}",
        ).hex[:12]

        return f"{now_ms:013d}-{_ID_NAMESPACE}-{_ID_SEQUENCE:06d}-{digest}"


class _PluginContextImpl:
    """Concrete PluginContext that mediates access from a plugin to the core.

    All memory writes are attributed with source=f"plugin:{plugin_name}".
    Read operations return empty results if no active stack.
    """

    def __init__(self, entity: Entity, plugin_name: str) -> None:
        self._entity = entity
        self._plugin_name = plugin_name

    @property
    def core_id(self) -> str:
        return self._entity.core_id

    @property
    def active_stack_id(self) -> Optional[str]:
        stack = self._entity.active_stack
        return stack.stack_id if stack else None

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.episode(
            objective,
            outcome,
            lessons=lessons,
            repeat=repeat,
            avoid=avoid,
            tags=tags,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def belief(
        self,
        statement: str,
        *,
        belief_type: str = "fact",
        confidence: float = 0.8,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.belief(
            statement,
            type=belief_type,
            confidence=confidence,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.value(
            name,
            statement,
            priority=priority,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.goal(
            title,
            description=description,
            goal_type=goal_type,
            priority=priority,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def note(
        self,
        content: str,
        *,
        note_type: str = "note",
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.note(
            content,
            type=note_type,
            tags=tags,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def relationship(
        self,
        other_entity_id: str,
        *,
        trust_level: Optional[float] = None,
        interaction_type: Optional[str] = None,
        notes: Optional[str] = None,
        entity_type: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.relationship(
            other_entity_id,
            trust_level=trust_level,
            notes=notes,
            interaction_type=interaction_type,
            entity_type=entity_type,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            source_type=source_type,
        )

    def drive(
        self,
        drive_type: str,
        *,
        intensity: float = 0.5,
        focus_areas: Optional[list[str]] = None,
        decay_hours: int = 24,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.drive(
            drive_type,
            intensity=intensity,
            focus_areas=focus_areas,
            decay_hours=decay_hours,
            derived_from=derived_from,
            source=f"plugin:{self._plugin_name}",
            context=context,
            source_type=source_type,
        )

    def raw(
        self,
        content: str,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.raw(
            content,
            source=f"plugin:{self._plugin_name}",
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.search(query, limit=limit, record_types=record_types, context=context)

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> list[Relationship]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.get_relationships(
            entity_id=entity_id, entity_type=entity_type, min_trust=min_trust
        )

    def get_goals(
        self,
        *,
        status: Optional[str] = None,
        context: Optional[str] = None,
    ) -> list[Goal]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.get_goals(status=status, context=context)

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.trust_set(entity, domain, score, evidence=evidence)

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return self._entity.trust_get(entity, domain=domain)

    def get_data_dir(self) -> Path:
        data_dir = self._entity._data_dir / "plugins" / self._plugin_name / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_config(self, key: str, default: Any = None) -> Any:
        config = self._entity._plugin_configs.get(self._plugin_name, {})
        return config.get(key, default)

    def get_secret(self, key: str) -> Optional[str]:
        secrets = self._entity._plugin_secrets.get(self._plugin_name, {})
        return secrets.get(key)


class Entity:
    """The Core coordinator/bus — implements CoreProtocol.

    Entity manages stack composition, plugin lifecycle, model binding,
    and routes memory operations to the active stack with provenance
    enforcement.
    """

    def __init__(
        self,
        core_id: str,
        data_dir: Optional[Path] = None,
        plugin_fail_fast: Optional[bool] = None,
    ) -> None:
        self._core_id = core_id
        self._data_dir = data_dir or get_kernle_home()
        self._model: Optional[ModelProtocol] = None
        self._stacks: dict[str, StackProtocol] = {}  # stack_id -> stack
        self._stack_aliases: dict[str, Optional[str]] = {}  # stack_id -> alias (cosmetic)
        self._active_stack_id: Optional[str] = None
        self._plugins: dict[str, PluginProtocol] = {}
        self._plugin_contexts: dict[str, _PluginContextImpl] = {}
        self._plugin_tools: dict[str, list[ToolDefinition]] = {}
        self._plugin_health: dict[str, PluginHealth] = {}
        self._plugin_configs: dict[str, dict[str, Any]] = {}
        self._plugin_secrets: dict[str, dict[str, str]] = {}
        self._plugin_fail_fast = (
            bool(plugin_fail_fast)
            if plugin_fail_fast is not None
            else _env_truthy(os.getenv("KERNLE_PLUGIN_FAIL_FAST"))
        )
        self._restored_binding: Optional[Binding] = None

    # ---- Core Properties ----

    @property
    def core_id(self) -> str:
        return self._core_id

    @property
    def model(self) -> Optional[ModelProtocol]:
        return self._model

    def set_model(self, model: ModelProtocol) -> None:
        self._model = model
        inference = self._get_inference_service()
        for stack in self._stacks.values():
            stack.on_model_changed(inference)

    @property
    def active_stack(self) -> Optional[StackProtocol]:
        if self._active_stack_id:
            return self._stacks.get(self._active_stack_id)
        return None

    @property
    def stacks(self) -> dict[str, StackInfo]:
        result: dict[str, StackInfo] = {}
        for sid, stack in self._stacks.items():
            result[sid] = StackInfo(
                stack_id=sid,
                alias=self._stack_aliases.get(sid),
                schema_version=stack.schema_version,
                stats=stack.get_stats(),
                is_active=sid == self._active_stack_id,
            )
        return result

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        result: dict[str, PluginInfo] = {}
        for name, plugin in self._plugins.items():
            result[name] = PluginInfo(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                capabilities=plugin.capabilities(),
                is_loaded=True,
            )
        return result

    # ---- Stack Management ----

    def attach_stack(
        self,
        stack: StackProtocol,
        *,
        alias: Optional[str] = None,
        set_active: bool = True,
    ) -> str:
        sid = stack.stack_id
        if sid in self._stacks:
            raise ValueError(f"Stack with stack_id '{sid}' is already attached")
        self._stacks[sid] = stack
        self._stack_aliases[sid] = alias
        stack.on_attach(self._core_id, self._get_inference_service())
        if set_active:
            self._active_stack_id = sid
        # Register already-loaded plugins with the new stack
        if hasattr(stack, "register_plugin"):
            for plugin_name in self._plugins:
                stack.register_plugin(plugin_name)
        return sid

    def detach_stack(self, stack_id: str) -> None:
        stack = self._stacks.pop(stack_id, None)
        if stack:
            stack.on_detach(self._core_id)
        self._stack_aliases.pop(stack_id, None)
        if self._active_stack_id == stack_id:
            self._active_stack_id = None

    def set_active_stack(self, stack_id: str) -> None:
        if stack_id not in self._stacks:
            raise ValueError(f"No stack with stack_id '{stack_id}'")
        self._active_stack_id = stack_id

    # ---- Plugin Management ----

    def plugin_health(self, name: str) -> Optional[PluginHealth]:
        return self._plugin_health.get(name)

    def load_plugin(
        self,
        plugin: PluginProtocol,
        *,
        subparsers: Any = None,
        fail_fast: Optional[bool] = None,
    ) -> None:
        from kernle.protocols import PROTOCOL_VERSION

        plugin_pv = getattr(plugin, "protocol_version", None)
        if plugin_pv is not None and plugin_pv > PROTOCOL_VERSION:
            raise ValueError(
                f"Plugin '{plugin.name}' requires protocol version {plugin_pv}, "
                f"but this core supports version {PROTOCOL_VERSION}."
            )
        elif plugin_pv is not None and plugin_pv < PROTOCOL_VERSION:
            logger.warning(
                "Plugin '%s' uses protocol version %d (current: %d).",
                plugin.name,
                plugin_pv,
                PROTOCOL_VERSION,
            )
        plugin_name = plugin.name
        fail_fast = self._plugin_fail_fast if fail_fast is None else fail_fast
        if plugin_name in self._plugins and self._plugin_health.get(plugin_name):
            # Keep the latest health state only from a prior load cycle.
            self._plugin_health.pop(plugin_name, None)

        context = _PluginContextImpl(self, plugin.name)
        plugin.activate(context)
        self._plugins[plugin.name] = plugin
        self._plugin_contexts[plugin.name] = context
        self._plugin_health[plugin.name] = PluginHealth(healthy=True, message="plugin loaded")

        def _rollback_loaded_plugin() -> None:
            self._plugins.pop(plugin.name, None)
            self._plugin_contexts.pop(plugin.name, None)
            self._plugin_tools.pop(plugin.name, None)
            self._plugin_health.pop(plugin.name, None)
            if self.active_stack and hasattr(self.active_stack, "unregister_plugin"):
                try:
                    self.active_stack.unregister_plugin(plugin.name)
                except Exception:
                    logger.debug(
                        "Plugin '%s' unregister during rollback failed", plugin.name, exc_info=True
                    )
            try:
                plugin.deactivate()
            except Exception:
                logger.debug(
                    "Plugin '%s' deactivation failed during rollback",
                    plugin.name,
                    exc_info=True,
                )

        # Register plugin with active stack for provenance bypass trust
        if self.active_stack and hasattr(self.active_stack, "register_plugin"):
            self.active_stack.register_plugin(plugin.name)

        # Register tools
        try:
            tools = plugin.register_tools()
            if tools:
                self._plugin_tools[plugin.name] = tools
        except Exception as e:
            message = f"Plugin '{plugin_name}' tool registration failed: {e}"
            logger.warning(message, exc_info=True)
            self._plugin_health[plugin_name] = PluginHealth(
                healthy=False,
                message=message,
            )
            if fail_fast:
                _rollback_loaded_plugin()
                raise RuntimeError(message) from e

        # Register CLI commands if subparsers provided
        if subparsers is not None:
            try:
                plugin.register_cli(subparsers)
            except Exception as e:
                message = f"Plugin '{plugin_name}' CLI registration failed: {e}"
                logger.warning(message, exc_info=True)
                self._plugin_health[plugin_name] = PluginHealth(
                    healthy=False,
                    message=message,
                )
                if fail_fast:
                    _rollback_loaded_plugin()
                    raise RuntimeError(message) from e

    def unload_plugin(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.deactivate()
        self._plugin_contexts.pop(name, None)
        self._plugin_tools.pop(name, None)
        self._plugin_health.pop(name, None)
        # Unregister plugin from active stack
        if self.active_stack and hasattr(self.active_stack, "unregister_plugin"):
            self.active_stack.unregister_plugin(name)

    def discover_plugins(self) -> list[PluginInfo]:
        discovered = discover_plugins()
        loaded_names = set(self._plugins.keys())
        result: list[PluginInfo] = []
        for comp in discovered:
            result.append(
                PluginInfo(
                    name=comp.name,
                    version=comp.dist_version or "unknown",
                    description="",
                    is_loaded=comp.name in loaded_names,
                )
            )
        return result

    def get_all_plugin_tools(self) -> list[ToolDefinition]:
        """Get all tools from all loaded plugins."""
        tools: list[ToolDefinition] = []
        for plugin_tools in self._plugin_tools.values():
            tools.extend(plugin_tools)
        return tools

    # ---- Routed Memory Operations (Provenance Enforcement) ----

    def _require_active_stack(self) -> StackProtocol:
        stack = self.active_stack
        if stack is None:
            raise NoActiveStackError("No active stack attached")
        return stack

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        episode_id = _generate_id()
        ep = Episode(
            id=episode_id,
            stack_id=stack.stack_id,
            objective=objective,
            outcome=outcome,
            outcome_type=infer_outcome_type(outcome),
            lessons=lessons,
            repeat=repeat,
            avoid=avoid,
            tags=tags or ["manual"],
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
            source_type=_normalize_source_type(source_type).value,
            derived_from=build_derived_from(derived_from, source),
            context=context,
            context_tags=context_tags,
        )
        if source:
            ep.source_entity = source
        else:
            ep.source_entity = f"core:{self._core_id}"
        stack.save_episode(ep)
        log_save(
            stack.stack_id,
            memory_type="episode",
            memory_id=episode_id,
            summary=objective[:50],
        )
        return episode_id

    def belief(
        self,
        statement: str,
        *,
        type: str = "fact",
        confidence: float = 0.8,
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        b = Belief(
            id=_generate_id(),
            stack_id=stack.stack_id,
            statement=statement,
            belief_type=normalize_belief_type(type),
            confidence=clamp_confidence(confidence),
            created_at=datetime.now(timezone.utc),
            is_protected=foundational,
            source_type=_normalize_source_type(source_type).value,
            derived_from=build_derived_from(derived_from, source),
            context=context,
            context_tags=context_tags,
        )
        if source:
            b.source_entity = source
        else:
            b.source_entity = f"core:{self._core_id}"
        return stack.save_belief(b)

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        type: str = "core_value",
        foundational: bool = False,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        v = Value(
            id=_generate_id(),
            stack_id=stack.stack_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            is_protected=foundational,
            source_type=_normalize_source_type(source_type).value,
            derived_from=build_derived_from(derived_from, source),
            context=context,
            context_tags=context_tags,
        )
        if source:
            v.source_entity = source
        else:
            v.source_entity = f"core:{self._core_id}"
        return stack.save_value(v)

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        validate_goal_type(goal_type)
        is_protected = goal_type in ("aspiration", "commitment")
        goal_id = _generate_id()
        g = Goal(
            id=goal_id,
            stack_id=stack.stack_id,
            title=title,
            description=description or title,
            goal_type=goal_type,
            priority=priority,
            status="active",
            is_protected=is_protected,
            created_at=datetime.now(timezone.utc),
            source_type=_normalize_source_type(source_type).value,
            derived_from=build_derived_from(derived_from, source),
            context=context,
            context_tags=context_tags,
        )
        if source:
            g.source_entity = source
        else:
            g.source_entity = f"core:{self._core_id}"
        stack.save_goal(g)
        if is_protected:
            stack.protect_memory("goal", goal_id, protected=True)
        return goal_id

    def note(
        self,
        content: str,
        *,
        type: str = "note",
        speaker: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[list[str]] = None,
        protect: bool = False,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        note_type = normalize_note_type(type)
        formatted = format_note_content(content, note_type, speaker=speaker, reason=reason)
        n = Note(
            id=_generate_id(),
            stack_id=stack.stack_id,
            content=formatted,
            note_type=note_type,
            speaker=speaker,
            reason=reason,
            tags=tags or [],
            created_at=datetime.now(timezone.utc),
            is_protected=protect,
            source_type=_normalize_source_type(source_type).value,
            derived_from=build_derived_from(derived_from, source),
            context=context,
            context_tags=context_tags,
        )
        if source:
            n.source_entity = source
        else:
            n.source_entity = f"core:{self._core_id}"
        return stack.save_note(n)

    def drive(
        self,
        drive_type: str,
        *,
        intensity: float = 0.5,
        focus_areas: Optional[list[str]] = None,
        decay_hours: int = 24,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        validate_drive_type(drive_type)
        now = datetime.now(timezone.utc)

        existing = stack.get_drive(drive_type)
        if existing:
            # Update in place — preserves ID, version chain, created_at
            existing.intensity = clamp_intensity(intensity)
            existing.updated_at = now
            if focus_areas is not None:
                existing.focus_areas = focus_areas
            if context is not None:
                existing.context = context
            if context_tags is not None:
                existing.context_tags = context_tags
            # Provenance: only overwrite when caller explicitly provides
            if source_type is not None:
                existing.source_type = _normalize_source_type(source_type).value
            if derived_from is not None:
                existing.derived_from = build_derived_from(derived_from, source)
            elif source is not None:
                existing.derived_from = build_derived_from(existing.derived_from, source)
            if source is not None:
                existing.source_entity = source
            if not stack.update_drive_atomic(existing):
                raise RuntimeError(f"Drive {existing.id} disappeared between get and update")
            return existing.id
        else:
            # Create new
            source_entity = source or f"core:{self._core_id}"
            st = _normalize_source_type(source_type).value
            df = build_derived_from(derived_from, source)
            d = Drive(
                id=_generate_id(),
                stack_id=stack.stack_id,
                drive_type=drive_type,
                intensity=clamp_intensity(intensity),
                focus_areas=focus_areas or [],
                created_at=now,
                updated_at=now,
                source_type=st,
                derived_from=df,
                context=context,
                context_tags=context_tags,
            )
            d.source_entity = source_entity
            return stack.save_drive(d)

    def relationship(
        self,
        other_stack_id: str,
        *,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        stack = self._require_active_stack()
        now = datetime.now(timezone.utc)

        existing = stack.get_relationship(other_stack_id)
        if existing:
            # Update in place — preserves ID, version chain, history
            if trust_level is not None:
                sentiment = max(-1.0, min(1.0, (trust_level * 2) - 1))
                existing.sentiment = sentiment
            if notes is not None:
                existing.notes = notes
            if interaction_type is not None:
                existing.relationship_type = interaction_type
            if entity_type is not None:
                existing.entity_type = entity_type
            existing.interaction_count = (existing.interaction_count or 0) + 1
            existing.last_interaction = now
            # Provenance: only overwrite when caller explicitly provides
            if source_type is not None:
                existing.source_type = _normalize_source_type(source_type).value
            if derived_from is not None:
                existing.derived_from = build_derived_from(derived_from, source)
            elif source is not None:
                existing.derived_from = build_derived_from(existing.derived_from, source)
            if source is not None:
                existing.source_entity = source
            if not stack.update_relationship_atomic(existing):
                raise RuntimeError(f"Relationship {existing.id} disappeared between get and update")
            return existing.id
        else:
            # Create new
            source_entity = source or f"core:{self._core_id}"
            st = _normalize_source_type(source_type).value
            df = build_derived_from(derived_from, source)
            sentiment = ((trust_level * 2) - 1) if trust_level is not None else 0.0
            sentiment = max(-1.0, min(1.0, sentiment))
            r = Relationship(
                id=_generate_id(),
                stack_id=stack.stack_id,
                entity_name=other_stack_id,
                entity_type=entity_type or "person",
                relationship_type=interaction_type or "interaction",
                notes=notes,
                sentiment=sentiment,
                interaction_count=1,
                last_interaction=now,
                created_at=now,
                source_type=st,
                derived_from=df,
            )
            r.source_entity = source_entity
            return stack.save_relationship(r)

    def raw(
        self,
        content: str,
        *,
        source: Optional[str] = None,
    ) -> str:
        stack = self._require_active_stack()
        r = RawEntry(
            id=_generate_id(),
            stack_id=stack.stack_id,
            blob=content,
            captured_at=datetime.now(timezone.utc),
            source=source or f"core:{self._core_id}",
        )
        return stack.save_raw(r)

    # ---- Routed Search & Load ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]:
        stack = self._require_active_stack()
        return stack.search(query, limit=limit, record_types=record_types, context=context)

    def load(
        self,
        *,
        token_budget: int = 8000,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        stack = self._require_active_stack()
        result = stack.load(token_budget=token_budget, context=context)
        for plugin in self._plugins.values():
            try:
                plugin.on_load(result)
            except Exception:
                logger.exception("Plugin %s failed on_load", plugin.name)
        return result

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "core_id": self._core_id,
            "model": self._model.model_id if self._model else None,
            "stacks": {},
            "plugins": {},
        }
        for sid, stack in self._stacks.items():
            result["stacks"][sid] = {
                "stack_id": sid,
                "alias": self._stack_aliases.get(sid),
                "active": sid == self._active_stack_id,
                "stats": stack.get_stats(),
            }
        for name, plugin in self._plugins.items():
            try:
                stored_health = self._plugin_health.get(name)
                if stored_health is not None and not stored_health.healthy:
                    health = stored_health
                else:
                    health = plugin.health_check()
            except Exception as exc:
                logger.debug(
                    "Swallowed %s in plugin '%s' health_check: %s",
                    type(exc).__name__,
                    name,
                    exc,
                    exc_info=True,
                )
                health = PluginHealth(healthy=False, message="health_check failed")
            result["plugins"][name] = {
                "version": plugin.version,
                "health": {"healthy": health.healthy, "message": health.message},
            }
        for plugin in self._plugins.values():
            try:
                plugin.on_status(result)
            except Exception:
                logger.exception("Plugin %s failed on_status", plugin.name)
        return result

    # ---- Routed Trust ----

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> str:
        stack = self._require_active_stack()
        assessment = TrustAssessment(
            id=_generate_id(),
            stack_id=stack.stack_id,
            entity=entity,
            dimensions={domain: {"score": score}},
            evidence_episode_ids=[evidence] if evidence else None,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        return stack.save_trust_assessment(assessment)

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]:
        stack = self._require_active_stack()
        return stack.get_trust_assessments(entity_id=entity, domain=domain)

    def trust_list(
        self,
        *,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[TrustAssessment]:
        stack = self._require_active_stack()
        assessments = stack.get_trust_assessments(domain=domain)
        if min_score is not None:
            filtered = []
            for a in assessments:
                for dim_data in a.dimensions.values():
                    if isinstance(dim_data, dict) and dim_data.get("score", 0) >= min_score:
                        filtered.append(a)
                        break
            return filtered
        return assessments

    # ---- Routed Memory Control ----

    def weaken(
        self,
        memory_type: str,
        memory_id: str,
        amount: float,
        *,
        reason: Optional[str] = None,
    ) -> bool:
        """Reduce a memory's strength by a given amount.

        Args:
            memory_type: Type of memory (episode, belief, etc.)
            memory_id: ID of the memory
            amount: Amount to reduce strength by (positive value)
            reason: Optional reason for weakening

        Returns:
            True if weakened, False if not found or protected
        """
        stack = self._require_active_stack()
        success = stack.weaken_memory(memory_type, memory_id, amount)
        if success:
            stack.log_audit(
                memory_type,
                memory_id,
                "weaken",
                actor=f"core:{self._core_id}",
                details={"amount": amount, "reason": reason},
            )
        return success

    def forget(
        self,
        memory_type: str,
        memory_id: str,
        reason: str,
    ) -> bool:
        """Forget a memory (set strength to 0.0).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Why this memory is being forgotten

        Returns:
            True if forgotten, False if not found or protected
        """
        stack = self._require_active_stack()
        return stack.forget_memory(memory_type, memory_id, reason)

    def recover(
        self,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        """Recover a forgotten memory (restore strength to 0.2).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        stack = self._require_active_stack()
        return stack.recover_memory(memory_type, memory_id)

    def verify(
        self,
        memory_type: str,
        memory_id: str,
        *,
        evidence: Optional[str] = None,
    ) -> bool:
        """Verify a memory: boost strength and increment verification count.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            evidence: Optional evidence supporting the verification

        Returns:
            True if verified, False if not found
        """
        stack = self._require_active_stack()
        success = stack.verify_memory(memory_type, memory_id)
        if success:
            stack.log_audit(
                memory_type,
                memory_id,
                "verify",
                actor=f"core:{self._core_id}",
                details={"evidence": evidence} if evidence else None,
            )
        return success

    def get_ungrounded_memories(self) -> list[tuple]:
        """Find memories where all source refs have strength 0.0 or don't exist.

        Returns:
            List of (memory_type, memory_id, [source_refs]) tuples
        """
        stack = self._require_active_stack()
        return stack.get_ungrounded_memories()

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> list[tuple]:
        """Find all memories that cite 'type:id' in their derived_from.

        Args:
            memory_type: Type of the source memory
            memory_id: ID of the source memory

        Returns:
            List of (child_memory_type, child_memory_id) tuples
        """
        stack = self._require_active_stack()
        return stack.get_memories_derived_from(memory_type, memory_id)

    def protect(
        self,
        memory_type: str,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        """Protect or unprotect a memory from forgetting/decay.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if not found
        """
        stack = self._require_active_stack()
        return stack.protect_memory(memory_type, memory_id, protected)

    # ---- Memory Processing ----

    def process(
        self,
        transition: Optional[str] = None,
        *,
        force: bool = False,
        strict: bool = False,
        allow_no_inference_override: bool = False,
        auto_promote: bool = False,
        batch_size: Optional[int] = None,
    ) -> list:
        """Run memory processing sessions.

        By default, creates suggestions for review rather than directly
        promoting memories. Set auto_promote=True to directly write memories
        (opt-in only).

        When no model is bound (inference unavailable), identity-layer
        transitions are blocked by the no-inference safety policy.
        Values can never be created without inference. Other identity
        layers require explicit override with force=True and
        allow_no_inference_override=True.

        Args:
            transition: Specific layer transition to process (None = check all)
            force: Process even if triggers aren't met
            strict: Raise on configuration load/parse errors instead of using defaults.
            allow_no_inference_override: Allow identity-layer writes without
                inference (except values). Only effective with force=True.
            auto_promote: If True, directly write memories. If False (default),
                create suggestions for review.
            batch_size: Override the per-transition batch size (None = use config).

        Returns:
            List of ProcessingResult for each transition that ran
        """
        stack = self._require_active_stack()
        inference = self._get_inference_service()
        inference_available = inference is not None

        if not inference_available:
            logger.warning("Processing without inference — identity-layer writes will be gated")

        from kernle.processing import MemoryProcessor, PromotionGateConfig

        # Load promotion gate config from stack settings
        promotion_gates = PromotionGateConfig()
        try:
            bme = stack.get_stack_setting("promotion_gate_belief_min_evidence")
            if bme is not None:
                promotion_gates.belief_min_evidence = int(bme)
            bmc = stack.get_stack_setting("promotion_gate_belief_min_confidence")
            if bmc is not None:
                promotion_gates.belief_min_confidence = float(bmc)
            vme = stack.get_stack_setting("promotion_gate_value_min_evidence")
            if vme is not None:
                promotion_gates.value_min_evidence = int(vme)
            vrp = stack.get_stack_setting("promotion_gate_value_requires_protection")
            if vrp is not None:
                promotion_gates.value_requires_protection = vrp == "true"
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                "Using default promotion gates due to stack setting parse failure on %s: %s",
                stack.stack_id,
                exc,
                exc_info=True,
            )

        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id=self._core_id,
            inference_available=inference_available,
            auto_promote=auto_promote,
            promotion_gates=promotion_gates,
        )

        # Load any saved config from the stack
        try:
            saved_configs = stack.get_processing_config()
            for cfg_dict in saved_configs:
                from kernle.processing import LayerConfig

                lc = LayerConfig(
                    layer_transition=cfg_dict["layer_transition"],
                    enabled=cfg_dict.get("enabled", True),
                    model_id=cfg_dict.get("model_id"),
                    quantity_threshold=cfg_dict.get("quantity_threshold") or 10,
                    valence_threshold=cfg_dict.get("valence_threshold") or 3.0,
                    time_threshold_hours=cfg_dict.get("time_threshold_hours") or 24,
                    batch_size=cfg_dict.get("batch_size") or 10,
                    max_sessions_per_day=cfg_dict.get("max_sessions_per_day") or 10,
                )
                processor.update_config(lc.layer_transition, lc)
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                "Using default processing config due to stack config parse failure on %s: %s",
                stack.stack_id,
                exc,
                exc_info=True,
            )

        return processor.process(
            transition,
            force=force,
            allow_no_inference_override=allow_no_inference_override,
            auto_promote=auto_promote,
            batch_size=batch_size,
        )

    # ---- Routed Sync ----

    def sync(self) -> SyncResult:
        stack = self._require_active_stack()
        return stack.sync()

    def checkpoint(self, message: str = "") -> str:
        self._require_active_stack()
        checkpoint_dir = self._data_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:8]
        cp_id = f"{self._core_id}_{ts}_{suffix}"
        cp_path = checkpoint_dir / f"{cp_id}.json"
        binding = self.get_binding()
        data = {
            "schema_version": 1,
            "checkpoint_id": cp_id,
            "message": message,
            "binding": {
                "core_id": binding.core_id,
                "model_config": binding.model_config,
                "stacks": binding.stacks,
                "active_stack_id": binding.active_stack_id,
                "plugins": binding.plugins,
            },
            "created_at": _now_iso(),
        }
        cp_path.write_text(json.dumps(data, indent=2))
        self._cleanup_old_checkpoints(checkpoint_dir)
        return cp_id

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep: int = 10) -> None:
        """Remove old checkpoints, keeping the most recent `keep` per core_id."""
        try:
            files = sorted(
                checkpoint_dir.glob(f"{self._core_id}_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError as exc:
            logger.warning("Failed to list checkpoints for cleanup: %s", exc)
            return
        to_remove = files[:-keep] if len(files) > keep else []
        for f in to_remove:
            try:
                f.unlink()
                logger.debug("Removed old checkpoint: %s", f.name)
            except OSError as exc:
                logger.warning("Failed to remove old checkpoint %s: %s", f.name, exc)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> Entity:
        """Restore an Entity from a checkpoint file.

        Reads checkpoint JSON, validates schema version, extracts
        the embedded binding, and delegates to from_binding(Binding)
        for full rehydration of stacks, plugins, and model.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.

        Returns:
            A rehydrated Entity instance.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            ValueError: If JSON is invalid, schema_version is unsupported,
                       or binding key is missing.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            data = json.loads(checkpoint_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in checkpoint: {exc}") from exc

        schema_version = data.get("schema_version")
        if schema_version is None:
            logger.warning(
                "Checkpoint %s has no schema_version — treating as v1",
                checkpoint_path.name,
            )
        elif schema_version > 1:
            raise ValueError(
                f"Unsupported checkpoint schema_version {schema_version} "
                f"(this code supports version 1)"
            )

        if "binding" not in data:
            raise ValueError(f"Checkpoint missing 'binding' key: {checkpoint_path}")

        binding_data = data["binding"]
        if not isinstance(binding_data, dict):
            raise ValueError(
                f"Checkpoint 'binding' must be a dict, got {type(binding_data).__name__}"
            )
        if "core_id" not in binding_data:
            raise ValueError("Checkpoint binding missing required 'core_id' field")
        binding = Binding(
            core_id=binding_data["core_id"],
            model_config=binding_data.get("model_config", {}),
            stacks=binding_data.get("stacks", {}),
            active_stack_id=binding_data.get("active_stack_id"),
            plugins=binding_data.get("plugins", []),
        )

        # Infer data_dir from checkpoint path structure
        data_dir = None
        if checkpoint_path.parent.name == "checkpoints":
            data_dir = checkpoint_path.parent.parent

        return cls.from_binding(binding, data_dir=data_dir)

    # ---- Binding Management ----

    def get_binding(self) -> Binding:
        stack_map: dict[str, Optional[str]] = {}
        for sid in self._stacks:
            stack_map[sid] = self._stack_aliases.get(sid)
        return Binding(
            core_id=self._core_id,
            model_config=(
                {"provider": self._model.capabilities.provider, "model_id": self._model.model_id}
                if self._model
                else {}
            ),
            stacks=stack_map,
            active_stack_id=self._active_stack_id,
            plugins=list(self._plugins.keys()),
            created_at=datetime.now(timezone.utc),
        )

    @classmethod
    def from_binding(cls, binding: Binding, *, data_dir: Optional[Path] = None) -> Entity:
        entity = cls(core_id=binding.core_id, data_dir=data_dir)
        entity._restored_binding = binding

        # Rehydrate stack instances from saved stack_id → alias map.
        for sid, alias in (binding.stacks or {}).items():
            try:
                from kernle.stack import SQLiteStack

                stack = SQLiteStack(stack_id=sid, enforce_provenance=False)
            except Exception as exc:
                logger.warning(
                    "Failed to instantiate stack '%s' for binding: %s",
                    sid,
                    exc,
                    exc_info=True,
                )
                continue
            try:
                entity.attach_stack(stack, alias=alias, set_active=False)
            except Exception as exc:
                logger.warning(
                    "Failed to attach stack '%s' from binding: %s", sid, exc, exc_info=True
                )

        if binding.active_stack_id is not None:
            if binding.active_stack_id in entity._stacks:
                try:
                    entity.set_active_stack(binding.active_stack_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to restore active stack '%s': %s",
                        binding.active_stack_id,
                        exc,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Binding active stack '%s' is missing after restore",
                    binding.active_stack_id,
                )

        # Restore plugins after stacks so plugin context is available.
        if binding.plugins:
            from kernle.discovery import discover_plugins, load_component

            discovered = {dc.name: dc for dc in discover_plugins()}
            for plugin_name in binding.plugins:
                discovered_plugin = discovered.get(plugin_name)
                if discovered_plugin is None:
                    logger.warning("Plugin '%s' from binding not found", plugin_name)
                    continue
                try:
                    plugin_cls = load_component(discovered_plugin)
                    entity.load_plugin(plugin_cls())
                except Exception as exc:
                    logger.warning(
                        "Failed to restore plugin '%s' from binding: %s",
                        plugin_name,
                        exc,
                        exc_info=True,
                    )

        # Restore model if the binding has sufficient metadata.
        if isinstance(binding.model_config, dict):
            provider = binding.model_config.get("provider")
            model_id = binding.model_config.get("model_id")
            if isinstance(provider, str) and isinstance(model_id, str) and provider and model_id:
                provider_key = provider.strip().lower()
                try:
                    if provider_key in {"claude", "anthropic"}:
                        from kernle.models.anthropic import AnthropicModel

                        entity.set_model(AnthropicModel(model_id=model_id))
                    elif provider_key == "openai":
                        from kernle.models.openai import OpenAIModel

                        entity.set_model(OpenAIModel(model_id=model_id))
                    elif provider_key == "ollama":
                        from kernle.models.ollama import OllamaModel

                        entity.set_model(OllamaModel(model_id=model_id))
                except Exception as exc:
                    logger.warning(
                        "Failed to restore model provider='%s' model_id='%s': %s",
                        provider_key,
                        model_id,
                        exc,
                        exc_info=True,
                    )

        return entity

    # ---- Internal Helpers ----

    def _get_inference_service(self) -> Optional[InferenceService]:
        """Create an InferenceService wrapping the current model.

        Returns None if no model is bound. Stacks and components
        degrade gracefully without it.
        """
        if self._model is None:
            return None
        from kernle.inference import create_inference_service

        return create_inference_service(self._model)
