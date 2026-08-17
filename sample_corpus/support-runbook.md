# Support Runbook

Operational procedures for the on-call engineer. Each entry states the
trigger, the diagnosis path, and the resolution. Escalate to the field team
if a procedure does not resolve within its stated time box.

## Many Robots Offline at Once

**Trigger:** more than 20 % of a fleet reports `link_state = offline` within
a two-minute span.

**Diagnosis.** This is a network problem until proven otherwise. A genuine
simultaneous hardware failure across many robots has never been observed in
production. Check, in order:

1. The access points serving the affected zone. Correlate the offline robots'
   last reported `position.zone` — if they cluster in one zone, it is that
   zone's radio.
2. The fleet's uplink to Fleet Control.
3. Only then, individual robots.

**Time box.** The offline autonomy window is 12 minutes for Meridian-3. If
you cannot restore the link inside that window the affected robots will
safe-park, which is safe but means a manual restart of throughput once the
link returns. Aim to have the radio diagnosed within 5 minutes.

**Resolution.** Once the link is restored the robots resume on their own.
Watch for `robot.link_restored` followed by `robot.safe_parked` on any robot
that did park. If a beacon stays amber for more than a minute after the link
returns, that robot needs a physical check.

## Single Robot Offline

**Trigger:** one robot reports `link_state = offline` for more than two
minutes.

**Diagnosis.** Read its last telemetry. The two fields that matter are
`position.zone` and `state_of_charge` at the last report.

- If the zone is a known radio dead spot, this is expected and self-resolving.
- If `state_of_charge` was below 25 % it was probably heading to a dock and
  the dock area coverage is the thing to check.
- If neither, dispatch someone to look at it after the offline autonomy
  window has elapsed — before that, it may simply be working through its
  buffer.

**Do not** dispatch replacement tasks to other robots for work already
buffered on the offline robot. Those tasks are on board and will complete;
duplicating them creates two robots heading to the same pick face.

## Dock Fault

**Trigger:** `dock.fault` webhook, or a bay reporting fault in the console.

**Diagnosis.** A single bay faulting is usually a seated-connection problem.
A whole bank of eight faulting at once points at the 3-phase feed for that
bank.

**Resolution.** Take the bay out of service in the console so Fleet Control
stops assigning it, then raise a field ticket. Do not power-cycle a bank
with a charge in progress — a Dockyard-2 bay never interrupts an in-progress
charge, and cutting power mid-charge is the one way to make it do so.

## Task Failure Rate Spike

**Trigger:** task failure rate above 5 % over fifteen minutes.

**Diagnosis.** Group the failures by robot and by zone before anything else.

- Concentrated on one robot: hardware. Take it out of the pool.
- Concentrated in one zone: layout or obstruction. Send someone to look.
- Spread evenly across the fleet: this is a Fleet Control or integration
  problem, not a robot problem. Check recent deploys first.

**Resolution.** For the integration case, the most common cause is a client
retrying task dispatch into the rate limit and interpreting `429` as a task
failure. Confirm by checking whether the failures carry
`error.code = "rate_limited"`.

## Emergency Stop Engaged

**Trigger:** `robot.estop` webhook.

**Diagnosis and resolution.** Send a human. E-stop recovery requires a
physical key twist at the robot and cannot be cleared from Fleet Control or
the API. There is no remote path, and any runbook step that suggests
otherwise is wrong.

Before clearing, the person at the robot must confirm the cause — the bumper
is a hardware interlock, so a bumper e-stop means the robot physically
contacted something.
