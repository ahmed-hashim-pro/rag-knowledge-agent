# Acme Robotics — Product Specifications

Acme Robotics builds autonomous mobile robots (AMRs) for warehouse fulfilment.
This document is the authoritative specification for hardware currently in
general availability. Figures are nominal at 20 °C unless stated otherwise.

## Meridian-3 Autonomous Mobile Robot

The Meridian-3 is our third-generation floor AMR, shipping since March 2024.
It replaces the Meridian-2 and is not mechanically compatible with Meridian-2
docking hardware.

### Physical and Payload

| Property | Value |
| --- | --- |
| Footprint | 720 mm × 540 mm |
| Height (deck) | 340 mm |
| Mass (unladen) | 68 kg |
| Rated payload | 120 kg |
| Absolute maximum payload | 150 kg (derated speed, see below) |
| Top speed (laden) | 1.8 m/s |
| Top speed above rated payload | 0.9 m/s |

Loading a Meridian-3 above its 120 kg rated payload does not fault the robot,
but the motion planner automatically halves the speed ceiling and the warranty
on the drive train is void above 150 kg.

### Power and Charging

The Meridian-3 carries a 48 V, 1.2 kWh LiFePO4 pack. Under a typical mixed
pick-and-transport duty cycle it runs for 7.5 hours between charges. Charging
from 20 % to 80 % takes 45 minutes on a Dockyard-2 station.

Robots return to charge automatically when the state of charge drops below
25 %. This threshold is configurable per fleet between 15 % and 40 %; below
15 % the robot cannot reliably reach a distant dock and the setting is
rejected.

### Offline Autonomy

Every Meridian-3 buffers its current task queue on board. If the robot loses
its network link to Fleet Control it continues executing the buffered queue
for up to **12 minutes**. This is the offline autonomy window.

At the end of the window, or when the buffered queue is exhausted — whichever
comes first — the robot performs a **safe-park**: it finishes its current
motion segment, pulls out of travel lanes to the nearest designated safe-park
zone, sets its status beacon to amber, and holds position until the link is
restored. A safe-parked robot does not accept new tasks and does not attempt
to return to a dock, because dock assignment requires Fleet Control.

The offline autonomy window is fixed in firmware and is not configurable. It
is deliberately shorter than the battery reserve so that a robot which loses
its link at low charge still has power to reach a dock once the link returns.

### Safety Systems

Two independent 2D LiDAR units cover a 360° field at deck height, backed by a
cliff sensor array and a physical bumper. The bumper is a hardware interlock:
contact cuts drive power without involving software. Emergency stop recovery
requires a physical key twist on the robot itself — it cannot be cleared from
Fleet Control or the API.

## Dockyard-2 Charging Station

The Dockyard-2 is the charging and staging station for Meridian-3 fleets.

### Capacity and Footprint

Each Dockyard-2 bay charges one robot. Bays gang together in banks of up to
eight from a single 3-phase feed. A bank of eight draws 11 kW peak.

Acme recommends provisioning one bay for every three robots in a
single-shift operation, and one bay for every two robots when running two or
more shifts.

### Firmware

Dockyard-2 firmware updates are pushed from Fleet Control and applied when a
bay is idle. A bay never interrupts an in-progress charge to update. Updates
to robots, by contrast, are applied at the next safe-park or dock event.
