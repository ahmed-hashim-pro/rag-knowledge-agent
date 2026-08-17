# Fleet Control API Reference

The Fleet Control API is the programmatic interface to an Acme Robotics
deployment. Base URL:

```
https://api.acmerobotics.example/v1
```

All requests and responses are JSON. All timestamps are RFC 3339 in UTC.

## Authentication

Pass a fleet API key in the `X-Acme-Key` header:

```
X-Acme-Key: fleet_live_<opaque>
```

Keys are scoped to a single fleet and carry one of three roles: `viewer`
(read-only), `operator` (read plus task dispatch), and `admin` (everything,
including firmware pins). A key can be rotated from the console; the previous
key stays valid for 24 hours after rotation to allow a clean cutover.

## Rate Limits

Rate limits are per fleet key, not per IP.

| Endpoint group | Limit |
| --- | --- |
| Telemetry reads | 600 requests/minute |
| Task dispatch | 120 requests/minute |
| Everything else | 60 requests/minute |

Exceeding a limit returns `429` with a `Retry-After` header in seconds.
Telemetry is the one group where polling hard is expected; if you find
yourself near the task-dispatch limit you should be batching (see
`POST /tasks:batch`).

## Fleet Telemetry

### GET /robots/{robot_id}/telemetry

Returns the current state of one robot. This is the endpoint to poll when you
need to know whether a robot is reachable.

```json
{
  "robot_id": "mrd3-0041",
  "model": "meridian-3",
  "link_state": "offline",
  "offline_seconds": 214,
  "state_of_charge": 0.62,
  "position": { "zone": "B", "x": 18.4, "y": 7.1 },
  "current_task": "task_01HZ...",
  "beacon": "amber",
  "updated_at": "2024-11-02T09:14:22Z"
}
```

`link_state` is one of:

- `online` — the robot has reported in within the last 5 seconds.
- `degraded` — last report is 5–30 seconds old. Usually radio congestion.
- `offline` — no report for more than 30 seconds. `offline_seconds` counts
  from the last successful report.

`offline_seconds` is the field to watch against the robot's offline autonomy
window. Fleet Control does not itself decide when a robot parks — the robot
makes that call on board — so `offline_seconds` crossing the window is your
signal that a safe-park has almost certainly happened. `beacon` will read
`amber` once the robot next reports in, if it did park.

### GET /robots

Lists robots in the fleet with the same shape as the single-robot endpoint.
Supports `?link_state=offline` to filter.

## Tasks

### POST /tasks

Dispatches one task. Returns `202` with a task ID. A task dispatched to a
robot that is `offline` is queued at Fleet Control, not on the robot, and is
delivered when the link returns.

### POST /tasks:batch

Dispatches up to 200 tasks in one request. Counts as a single request against
the task-dispatch rate limit. Partial success is possible: the response
contains a per-task `status` array.

### DELETE /tasks/{task_id}

Cancels a task. A task already buffered on board a robot that is currently
offline cannot be cancelled until the link returns; the call returns `409`
with `error.code = "robot_unreachable"`.

## Webhooks

Register endpoints in the console. Deliveries are signed with an HMAC-SHA256
signature in the `X-Acme-Signature` header; verify it before trusting the
body. Acme retries a failed delivery 5 times with exponential backoff over
roughly 15 minutes, then drops it.

Events:

| Event | Fires when |
| --- | --- |
| `robot.link_lost` | `link_state` transitions to `offline` |
| `robot.link_restored` | `link_state` transitions back to `online` |
| `robot.safe_parked` | A robot reports that it has safe-parked |
| `robot.estop` | Emergency stop engaged |
| `task.completed` | Task finished successfully |
| `task.failed` | Task ended in an error state |
| `dock.fault` | A Dockyard-2 bay reports a fault |

`robot.link_lost` fires immediately on the transition. `robot.safe_parked`
can only fire after the link is restored, because the robot has no way to
report while it is offline. Do not wait for `robot.safe_parked` to detect an
outage — use `robot.link_lost` plus the offline autonomy window.

## Error Codes

| HTTP | `error.code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Malformed body or parameters |
| 401 | `bad_key` | Missing or invalid API key |
| 403 | `insufficient_role` | Key role too low for this endpoint |
| 404 | `not_found` | Unknown robot, task, or dock |
| 409 | `robot_unreachable` | Robot is offline and the call needs it now |
| 409 | `estop_engaged` | Robot is in emergency stop |
| 429 | `rate_limited` | See `Retry-After` |
| 503 | `fleet_maintenance` | Fleet is in a maintenance window |
