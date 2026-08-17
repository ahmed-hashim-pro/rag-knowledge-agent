# Engineering Onboarding Guide

Welcome to Acme Robotics. This guide covers your first two weeks on the Fleet
Platform team. Read it end to end before your first standup.

## Day One

### Accounts and Access

Your manager files the access request before you start; by day one you should
have Okta, GitHub (`acme-robotics` org), and read access to the staging fleet
console. Production console access is granted after you complete the
safety briefing — not before, and there is no exception to this.

You will be issued two Fleet Control API keys: a `viewer` key for the staging
fleet on day one, and an `operator` key once you have shipped your first
change. Nobody on the platform team carries a production `admin` key by
default; those are checked out for the duration of a specific task.

### The Safety Briefing

A 90-minute session with the field team, run every Tuesday. It covers the
robot safety systems — the LiDAR envelope, the bumper interlock, and e-stop
recovery. The practical takeaway most engineers miss: **you cannot clear an
emergency stop from software**, so any tooling you build that assumes it can
recover a fleet remotely is wrong by construction.

## Development Environment

### Local Setup

Clone `acme-robotics/fleet-platform` and run `make bootstrap`. This brings up
a local Fleet Control against a simulated fleet — sixteen simulated
Meridian-3 robots and two simulated Dockyard-2 banks. The simulator models
the offline autonomy window faithfully, including safe-park behaviour, so you
can develop outage handling without a radio.

To simulate an outage locally:

```
make sim-outage ROBOT=mrd3-sim-04 DURATION=15m
```

A duration longer than the offline autonomy window will produce a real
safe-park in the simulator, which is the case most integration bugs hide in.
Test it.

### Staging Fleet

Staging is four physical Meridian-3 robots in the Building C test cell. It is
shared. Book it in the `#fleet-staging` channel before you run anything that
dispatches tasks, and never leave a robot safe-parked overnight — it will
still be there in the morning and the next person will lose an hour.

## Shipping Changes

### Review and Deploy

Two approvals for anything touching task dispatch, one for everything else.
Deploys go out continuously; there is no release train. The deploy pipeline
runs the integration suite against the simulator, not against staging, so a
green pipeline is not evidence that your change works on real hardware.

### Firmware Is Different

Robot firmware does not follow the software deploy path. Firmware is pinned
per fleet and rolled out deliberately, and a robot only applies an update at
its next safe-park or dock event. This means a firmware rollout can take
hours to reach a busy fleet, and robots in the same fleet will be running
different firmware during that window. Write integrations that tolerate it.

## On-Call

### Rotation

One-week rotations, handed over on Monday morning. You join the rotation in
your second month, shadowing for the first week.

### What Actually Pages

Pages come from fleet-level signals, not individual robots. A single robot
going offline is normal — radio coverage is imperfect and the robots are
designed for it. What pages is a threshold: more than 20 % of a fleet offline
simultaneously, any dock fault, or task failure rate above 5 % over fifteen
minutes.

The most common false alarm is a zone-wide radio dropout that reads as many
robots offline at once. Check the access point before you check the robots.
