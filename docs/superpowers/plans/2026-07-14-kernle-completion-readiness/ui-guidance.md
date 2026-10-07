# SDK, CLI, MCP, and Integration Guidance

There is no visual UI in the recommended alpha. User experience is the contract of commands, tools, Python calls, messages, and durable state transitions.

## Required States

Every retained workflow must specify:

- uninitialized and initialized;
- empty and populated;
- success and idempotent replay;
- invalid input;
- unauthenticated/omitted requester;
- owner-access denial with stable reason;
- storage conflict/partial failure;
- corrupt export/restore and recovery;
- migration required/failed;
- MCP capture/promotion success and failure.

## Output Rules

- Human output is concise and actionable.
- Machine surfaces carry stable reason/error codes and structured fields.
- Denials reveal no protected content.
- A partial or failed write is never worded as success.
- Destructive/repair operations require explicit authority and preview.
- Docs examples show exact installed commands and expected observable results.

## Surface Choice

Recommended alpha surface order:

1. canonical Python API;
2. MCP as the sole product host;
3. CLI only for init/status/diagnostics;
4. one export/restore path.

Other hosts, model-driven processing, and dashboard UI are post-alpha only after their own evidence packets.

Dashboard/web/mobile UX is deferred. If reintroduced, add authoritative state designs under [mockups/](mockups/) and workflow acceptance fixtures before implementation.
