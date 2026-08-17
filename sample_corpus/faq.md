# Frequently Asked Questions

Answers to the questions our support team and field engineers field most
often. Where an answer depends on a hardware figure, the specification
document is the authority.

## Fleet Operations

### What happens during a network outage?

Meridian-3 robots keep working from their on-board task buffer for the
duration of the offline autonomy window, then safe-park and wait. Nothing is
lost: buffered tasks that completed while offline are reported to Fleet
Control when the link returns, and tasks dispatched during the outage are
held at Fleet Control and delivered on reconnect.

From the API side, the sequence you will observe is:

1. `robot.link_lost` webhook fires, and `link_state` flips to `offline`.
2. `offline_seconds` climbs on each telemetry read.
3. Once `offline_seconds` passes the offline autonomy window, assume the
   robot has safe-parked. You cannot confirm this while it is offline.
4. On reconnect you get `robot.link_restored`, and — if it did park — a
   `robot.safe_parked` event and an amber `beacon`.

A safe-parked robot resumes on its own once the link is back and Fleet
Control acknowledges it. No manual intervention is required unless the
beacon stays amber for more than a minute after reconnect.

### Can I make robots keep working longer without a network?

No. The offline autonomy window is fixed in firmware and is not exposed as a
configuration option. Customers who need longer independent operation are
usually better served by fixing radio coverage in the affected zone; our
field team will do a survey on request.

### Why did my robot slow down after we increased load?

Almost certainly you crossed the rated payload. Above the rated figure the
motion planner halves the speed ceiling automatically. This is not a fault
and nothing is logged as an error, which is why it often looks mysterious.
Check the load, not the robot.

### How many charging bays do we need?

The specification gives the planning ratios by shift pattern. In practice,
customers who follow the two-shift ratio and still see robots queueing at
docks usually have their auto-return threshold set too high — every robot
heading back at 40 % state of charge creates artificial contention.

## API and Integration

### How should I poll for robot status?

Poll `GET /robots/{robot_id}/telemetry` — the telemetry rate limit is
deliberately generous because polling it is the expected pattern. For fleets
above roughly 30 robots, prefer `GET /robots` with a filter over a loop of
per-robot calls; it is one request against the same limit.

Better still, use webhooks for state transitions and poll only to fill in
detail. Webhook deliveries are signed; always verify the signature header
before acting on a body.

### My task cancellation returned 409. What now?

`robot_unreachable` means the task is already buffered on a robot that is
currently offline, so Fleet Control cannot recall it. Wait for
`robot.link_restored` and retry the cancellation. If the task has completed
in the meantime you will get a `404` instead, which is the expected outcome.

### Does rotating an API key break running integrations?

Not immediately. The previous key keeps working for 24 hours after rotation,
which is the window you have to roll the new key out. After that the old key
returns `401 bad_key`.

## Hardware

### Can I use our existing Meridian-2 docks?

No. Meridian-3 is not mechanically compatible with Meridian-2 docking
hardware. Dockyard-2 is the only supported station.

### How do I clear an emergency stop?

Physically, at the robot, with the key. E-stop recovery is deliberately not
available from Fleet Control or the API — it requires a human to have looked
at the robot.
