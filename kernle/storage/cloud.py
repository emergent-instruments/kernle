"""Cloud search client for Kernle storage.

Handles cloud credential loading and remote search operations.
Zero DB coupling — pure HTTP/credential logic.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from kernle.credentials import resolve_credentials

from .base import (
    Belief,
    Episode,
    Goal,
    Note,
    SearchResult,
    Value,
    parse_datetime,
)

logger = logging.getLogger(__name__)


class CloudClient:
    """Cloud search client that manages credentials and remote search.

    Args:
        stack_id: The stack identifier (used when constructing record objects).
        cloud_search_timeout: Default timeout for cloud search requests.
    """

    def __init__(self, stack_id: str, cloud_search_timeout: float = 3.0):
        self.stack_id = stack_id
        self.cloud_search_timeout = cloud_search_timeout
        self._cloud_credentials: Optional[Dict[str, str]] = None
        self._cloud_credentials_loaded: bool = False

    def _load_cloud_credentials(self) -> Optional[Dict[str, str]]:
        """Load cloud credentials via shared resolve_credentials().

        Results are cached after first call. Returns dict with
        'backend_url' and 'auth_token', or None if not configured.
        """
        if self._cloud_credentials_loaded:
            return self._cloud_credentials

        self._cloud_credentials_loaded = True
        resolved = resolve_credentials()
        backend_url = resolved["backend_url"]
        auth_token = resolved["auth_token"]

        if backend_url and auth_token:
            self._cloud_credentials = {
                "backend_url": backend_url,
                "auth_token": auth_token,
            }
        else:
            self._cloud_credentials = None

        return self._cloud_credentials

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available.

        Returns:
            True if backend_url and auth_token are configured.
        """
        creds = self._load_cloud_credentials()
        return creds is not None

    def cloud_health_check(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Test cloud backend connectivity.

        Args:
            timeout: Request timeout in seconds (default 3s)

        Returns:
            Dict with keys:
            - 'healthy': bool indicating if cloud is reachable
            - 'latency_ms': response time in milliseconds (if healthy)
            - 'error': error message (if not healthy)
        """
        import time

        creds = self._load_cloud_credentials()
        if not creds:
            return {
                "healthy": False,
                "error": "No cloud credentials configured",
            }

        try:
            import urllib.error
            import urllib.request

            url = f"{creds['backend_url']}/health"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {creds['auth_token']}",
                    "Content-Type": "application/json",
                },
                method="GET",
            )

            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.time() - start) * 1000
                if resp.status == 200:
                    return {
                        "healthy": True,
                        "latency_ms": round(latency_ms, 2),
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {resp.status}",
                    }
        except urllib.error.URLError as e:
            return {
                "healthy": False,
                "error": f"Connection failed: {e.reason}",
            }
        except Exception as e:
            logger.debug("Cloud health check failed: %s", e, exc_info=True)
            return {
                "healthy": False,
                "error": str(e),
            }

    def _cloud_search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[SearchResult]]:
        """Search memories via cloud backend.

        Args:
            query: Search query
            limit: Maximum results
            record_types: Filter by type (episode, note, belief, etc.)
            timeout: Request timeout in seconds (default: cloud_search_timeout)

        Returns:
            List of SearchResult, or None if cloud search failed/unavailable.
        """
        creds = self._load_cloud_credentials()
        if not creds:
            return None

        timeout = timeout or self.cloud_search_timeout

        try:
            import urllib.error
            import urllib.request

            url = f"{creds['backend_url']}/memories/search"
            payload = {
                "query": query,
                "limit": limit,
            }
            if record_types:
                # Map internal types to backend table names
                type_map = {
                    "episode": "episodes",
                    "note": "notes",
                    "belief": "beliefs",
                    "value": "values",
                    "goal": "goals",
                }
                payload["memory_types"] = [type_map.get(t, t) for t in record_types]

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {creds['auth_token']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    return self._parse_cloud_search_results(response_data)
                else:
                    logger.debug(f"Cloud search returned HTTP {resp.status}")
                    return None

        except urllib.error.URLError as e:
            logger.debug(f"Cloud search failed: {e.reason}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Cloud search response parse error: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.debug(f"Cloud search error: {e}", exc_info=True)
            return None

    def _parse_cloud_search_results(self, response_data: Dict[str, Any]) -> List[SearchResult]:
        """Parse cloud search response into SearchResult objects.

        Args:
            response_data: Response from /memories/search endpoint

        Returns:
            List of SearchResult objects
        """
        results = []
        cloud_results = response_data.get("results", [])

        # Map backend types to internal types
        type_map = {
            "episodes": "episode",
            "notes": "note",
            "beliefs": "belief",
            "values": "value",
            "goals": "goal",
        }

        for item in cloud_results:
            memory_type = type_map.get(item.get("memory_type"), item.get("memory_type"))
            metadata = item.get("metadata", {})

            # Create a minimal record object based on type
            record = self._create_record_from_cloud(memory_type, item, metadata)

            if record:
                results.append(
                    SearchResult(
                        record=record,
                        record_type=memory_type,
                        score=item.get("score", 1.0),
                    )
                )

        return results

    def _create_record_from_cloud(
        self, memory_type: str, item: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Optional[Any]:
        """Create a record object from cloud search result.

        Args:
            memory_type: Type of memory (episode, note, etc.)
            item: Search result item
            metadata: Additional metadata from result

        Returns:
            Record object or None
        """
        record_id = item.get("id", "")
        created_at = parse_datetime(item.get("created_at"))

        if memory_type == "episode":
            return Episode(
                id=record_id,
                stack_id=self.stack_id,
                objective=metadata.get("objective", item.get("content", "")),
                outcome=metadata.get("outcome", ""),
                outcome_type=metadata.get("outcome_type"),
                lessons=metadata.get("lessons_learned"),
                tags=metadata.get("tags"),
                created_at=created_at,
            )
        elif memory_type == "note":
            return Note(
                id=record_id,
                stack_id=self.stack_id,
                content=item.get("content", ""),
                note_type=metadata.get("note_type", "note"),
                tags=metadata.get("tags"),
                created_at=created_at,
            )
        elif memory_type == "belief":
            return Belief(
                id=record_id,
                stack_id=self.stack_id,
                statement=item.get("content", ""),
                belief_type=metadata.get("belief_type", "fact"),
                confidence=metadata.get("confidence", 0.8),
                created_at=created_at,
            )
        elif memory_type == "value":
            return Value(
                id=record_id,
                stack_id=self.stack_id,
                name=metadata.get("name", ""),
                statement=item.get("content", ""),
                priority=metadata.get("priority", 50),
                created_at=created_at,
            )
        elif memory_type == "goal":
            return Goal(
                id=record_id,
                stack_id=self.stack_id,
                title=metadata.get("title", item.get("content", "")),
                description=metadata.get("description"),
                priority=metadata.get("priority", "medium"),
                status=metadata.get("status", "active"),
                created_at=created_at,
            )

        return None
