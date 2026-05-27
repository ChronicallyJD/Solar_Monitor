"""
solar_monitor/scanner.py — BLE scanning, device resolution, and poll orchestration
====================================================================================
Coordinates the full polling cycle:
  1. BLE scan (PersistentScanner keeps radio on throughout)
  2. Device resolution: match configured devices to scan results
  3. Sequential BMS polling (BlueZ reliability constraint)
  4. Victron reading from accumulated advertisement payloads

BlueZ sequential polling rationale
------------------------------------
BlueZ on Linux serialises all GATT operations through a single D-Bus socket.
Firing more than ~2 concurrent BleakClient.connect() calls produces:
  - "org.bluez.Error.Failed: Operation already in progress"
  - "br-connection-canceled"
Polling BMS devices strictly one-at-a-time with a gap between each is slower
but produces far more reliable results across 5–10 devices.

Victron advertisement accumulation
-------------------------------------
Victron devices cycle through broadcasting multiple record types within a
single advertisement period.  PersistentScanner accumulates every distinct
payload seen per MAC so poll_all can give victron.read_victron_advertisement
a complete picture of what the device is broadcasting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from bleak import BleakScanner
from bleak.backends.device import BLEDevice

from .config import AppConfig
from .models import DeviceReading
from .jbd import JBD_NAME_KEYWORDS, read_jbd_device
from .victron import (
    VICTRON_MFR_ID, VICTRON_NAME_KEYWORDS, VICTRON_RECORD_TYPES,
    read_victron_advertisement,
)

log = logging.getLogger(__name__)

# ── Retry / timing constants ──────────────────────────────────────────────────

BMS_RETRIES      = 3    # total attempts per BMS device (including the first)
RETRY_DELAY      = 4.0  # seconds to wait between retry attempts
INTER_DEVICE_GAP = 1.0  # seconds between successive BMS connections

# Error substrings that indicate a transient BlueZ failure worth retrying
_TRANSIENT_ERRORS: tuple[str, ...] = (
    "operation already in progress",
    "br-connection-canceled",
    "connection attempt",
    "failed to connect",
    "not found",
    "removed from bluez",
    "le connection",
    "timed out",
    "payload too short",
    "",   # empty = asyncio.TimeoutError stringified
)


# ── Persistent BLE scanner ────────────────────────────────────────────────────

class PersistentScanner:
    """
    Long-lived BLE scanner that keeps the radio on throughout the poll cycle.

    Why persistent?
    ---------------
    ``BleakScanner.discover()`` stops the radio after its timeout, causing
    BlueZ to evict cached device objects.  Subsequent ``BleakClient(device)``
    calls then raise "device was not found / removed from BlueZ".
    Keeping the scanner running until all GATT connections are closed avoids
    this error entirely.

    Victron payload accumulation
    ----------------------------
    Victron devices cycle through advertising multiple record types within a
    single advertisement period (e.g. generic Solar Charger 0x01 AND device-
    specific Inverter 0x07 from the same MAC).  ``BleakScanner``'s callback
    fires once per received advertisement, so storing only the latest payload
    means we non-deterministically get whichever record arrived last.

    This scanner accumulates **all distinct payloads** per MAC so the reader
    can try every record type and choose the most informative one.
    """

    def __init__(self) -> None:
        self._seen:              dict[str, tuple]       = {}
        self._victron_payloads:  dict[str, list[bytes]] = {}
        self._scanner: Optional[BleakScanner]           = None

    def _cb(self, device: BLEDevice, adv_data) -> None:
        """BleakScanner detection callback — called once per advertisement received."""
        mac = device.address.upper()
        self._seen[mac] = (device, adv_data)

        # Accumulate distinct Victron Instant Readout payloads.
        # Filter out short VE.Smart networking beacons (Format A < 9 bytes)
        # and payloads with unknown record types.  VE.Smart beacons from other
        # devices on the VE.Smart network can arrive under a neighbour's MAC and
        # confuse the parser if not excluded.
        mfr = getattr(adv_data, "manufacturer_data", {}) or {}
        raw = mfr.get(VICTRON_MFR_ID)
        if not raw:
            return

        payloads = raw if isinstance(raw, list) else [raw]
        seen_set = {bytes(p) for p in self._victron_payloads.get(mac, [])}

        for p in payloads:
            pb = bytes(p)
            if not pb or pb in seen_set:
                continue
            # Must be a valid Instant Readout payload
            if pb[0] == 0x10:
                # Format A: must be >= 9 bytes (skips short 4-byte VE.Smart beacons)
                if len(pb) < 9:
                    continue
                record_type = pb[3] & 0x0F
            elif len(pb) >= 5:
                # Format B: record type is byte 0
                record_type = pb[0]
            else:
                continue
            # Only accumulate known record types
            if record_type not in VICTRON_RECORD_TYPES:
                continue
            self._victron_payloads.setdefault(mac, []).append(pb)
            seen_set.add(pb)

    async def scan(self, duration: float) -> dict[str, tuple]:
        """
        Start the scanner, wait *duration* seconds, and return a snapshot.

        The scanner continues running after this call returns so that:
        - BlueZ retains device references for BMS connections.
        - Victron advertisements keep arriving with fresh nonces.

        Call ``stop()`` explicitly once all BLE work is complete.
        """
        self._seen.clear()
        self._victron_payloads.clear()
        self._scanner = BleakScanner(detection_callback=self._cb)
        await self._scanner.start()
        await asyncio.sleep(duration)
        return dict(self._seen)

    def latest_adv(self, mac: str) -> Optional[tuple]:
        """Return the most recent ``(BLEDevice, adv_data)`` for *mac*, or None."""
        return self._seen.get(mac.upper())

    def victron_payloads(self, mac: str) -> list[bytes]:
        """
        Return all distinct Victron advertisement payloads accumulated for *mac*.

        The list may contain payloads of different record types collected over
        many advertisement cycles during the scan window.
        """
        return list(self._victron_payloads.get(mac.upper(), []))

    def snapshot(self) -> dict[str, tuple]:
        """Return a point-in-time copy of all seen devices."""
        return dict(self._seen)

    async def stop(self) -> None:
        """Stop the BLE scanner and release the radio."""
        if self._scanner:
            try:
                await self._scanner.stop()
            except Exception:
                pass
            self._scanner = None


# ── Device resolution ─────────────────────────────────────────────────────────

async def resolve_devices(cfg: AppConfig) -> tuple[list, list, PersistentScanner]:
    """
    Scan for BLE devices and resolve them against the current configuration.

    Returns:
        ``(jbd_pairs, mppt_triples, scanner)`` where:

        - ``jbd_pairs``: list of ``(BLEDevice, friendly_name, password)`` for
          BMS devices found in the scan, or ``(None, name, ident, password)``
          for configured devices that were not seen.

        - ``mppt_triples``: list of ``(BLEDevice, adv_data, name, enc_key)``
          for Victron devices found in the scan, or ``(None, None, name, ident)``
          for missing configured devices.

        - ``scanner``: the running PersistentScanner; caller must ``await
          scanner.stop()`` after all connections are finished.

    Device matching order:
      1. Configured devices matched by MAC (exact, case-insensitive).
      2. Configured BMS devices matched by BLE advertisement name.
      3. Auto-discovered devices matched by BLE name keyword patterns.
    """
    log.info("Scanning for BLE devices …")
    scanner = PersistentScanner()
    seen    = await scanner.scan(cfg.scan_timeout)
    log.info(f"Scan complete — {len(seen)} device(s) found")

    jbd_pairs    = []
    mppt_triples = []

    # ── Explicit BMS devices ──────────────────────────────────────────────────
    if not cfg.auto_discover_bms:
        for dc in cfg.bms_devices:
            entry = None
            if dc.mac:
                entry = seen.get(dc.mac.upper())
                if not entry:
                    log.warning(
                        f"  [BMS]  '{dc.name}' (MAC {dc.mac}) "
                        f"not seen in scan — will show as OFFLINE"
                    )
            elif dc.ble_name:
                ble_lower = dc.ble_name.lower()
                for _mac, (dev, adv) in seen.items():
                    if (dev.name or "").lower() == ble_lower:
                        entry = (dev, adv)
                        log.info(
                            f"  [BMS]  '{dc.name}' matched BLE name "
                            f"'{dc.ble_name}' -> {dev.address}"
                        )
                        break
                if not entry:
                    log.warning(
                        f"  [BMS]  '{dc.name}' (BLE name '{dc.ble_name}') "
                        f"not seen in scan — will show as OFFLINE"
                    )

            if entry:
                jbd_pairs.append((entry[0], dc.name, dc.password))
            else:
                placeholder = dc.mac or dc.ble_name or "??"
                jbd_pairs.append((None, dc.name, placeholder, dc.password))

    # ── Explicit Victron devices ──────────────────────────────────────────────
    # Pre-build key lookups so auto-discovery can attach keys too
    mppt_key_by_mac  = {
        dc.mac.upper(): dc.enc_key
        for dc in cfg.mppt_devices if dc.mac and dc.enc_key
    }
    mppt_key_by_name = {
        (dc.ble_name or "").lower(): dc.enc_key
        for dc in cfg.mppt_devices if dc.ble_name and dc.enc_key
    }

    if not cfg.auto_discover_mppt:
        for dc in cfg.mppt_devices:
            entry = None
            if dc.mac:
                entry = seen.get(dc.mac.upper())
            if entry is None and dc.ble_name:
                ble_lower = dc.ble_name.lower()
                for _mac, (dev, adv) in seen.items():
                    if (dev.name or "").lower() == ble_lower:
                        entry = (dev, adv)
                        log.info(
                            f"  [Victron] '{dc.name}' matched BLE name "
                            f"'{dc.ble_name}' -> {dev.address}"
                        )
                        break
            if entry:
                mppt_triples.append((entry[0], entry[1], dc.name, dc.enc_key))
            else:
                log.warning(
                    f"  [Victron] '{dc.name}' "
                    f"({dc.mac or dc.ble_name}) not seen in scan"
                )
                mppt_triples.append(
                    (None, None, dc.name, dc.mac or dc.ble_name or "unknown")
                )

    # ── Auto-discovery ────────────────────────────────────────────────────────
    explicit_macs  = ({dc.mac.upper() for dc in cfg.bms_devices  if dc.mac} |
                      {dc.mac.upper() for dc in cfg.mppt_devices if dc.mac})
    explicit_names = {(dc.ble_name or "").lower()
                      for dc in cfg.bms_devices if dc.ble_name}

    for mac, (dev, adv) in seen.items():
        if mac in explicit_macs:
            continue
        name_lower = (dev.name or "").lower()
        if name_lower in explicit_names:
            continue

        if cfg.auto_discover_bms and any(
            kw in name_lower for kw in JBD_NAME_KEYWORDS
        ):
            jbd_pairs.append((dev, dev.name or mac, None))
            log.info(f"  [BMS]  Auto-discovered: {dev.name or mac} ({mac})")

        elif cfg.auto_discover_mppt and (
            any(kw in name_lower for kw in VICTRON_NAME_KEYWORDS)
            or VICTRON_MFR_ID in (getattr(adv, "manufacturer_data", {}) or {})
        ):
            key = mppt_key_by_mac.get(mac) or mppt_key_by_name.get(name_lower)
            mppt_triples.append((dev, adv, dev.name or mac, key))
            log.info(f"  [Victron] Auto-discovered: {dev.name or mac} ({mac})"
                     + (" (key set)" if key else ""))

    log.info(
        f"Resolved {len(jbd_pairs)} BMS device(s), "
        f"{len(mppt_triples)} Victron device(s)"
    )
    return jbd_pairs, mppt_triples, scanner


# ── Poll orchestration ────────────────────────────────────────────────────────

async def poll_all(
    jbd_pairs: list,
    mppt_triples: list,
    scanner: PersistentScanner,
) -> tuple[list[DeviceReading], list[DeviceReading]]:
    """
    Read all devices and return ``(bms_readings, mppt_readings)``.

    BMS polling strategy
    --------------------
    BMS devices are polled **strictly sequentially** with a 1-second gap
    between each one.  Concurrent GATT connections on Linux/BlueZ are
    unreliable beyond 2 devices; sequential polling with retries gives far
    better success rates.

    Each device is attempted up to ``BMS_RETRIES`` times.  Only errors
    matching ``_TRANSIENT_ERRORS`` are retried; permanent errors (bad
    password, unsupported GATT service) stop immediately.

    Victron reading strategy
    ------------------------
    No GATT connection is needed for Victron devices.  All data comes from
    the BLE advertisements accumulated by the scanner.  Victron devices are
    read after all BMS connections complete (so the scanner is still running
    and delivering fresh nonces at the time of reading).

    The scanner is stopped after all Victron readings are complete.
    """

    async def _read_bms_with_retry(entry) -> DeviceReading:
        """Attempt to read a BMS device with transient-error retry."""
        if entry[0] is None:
            _, name, ident, *_ = entry
            return DeviceReading(
                address=ident, name=name, device_type="bms",
                timestamp=datetime.now().isoformat(timespec="seconds"),
                error="Device not found during scan",
            )

        dev, friendly, password = entry[0], entry[1], entry[2]
        result: Optional[DeviceReading] = None

        for attempt in range(BMS_RETRIES):
            if attempt > 0:
                log.info(
                    f"  [BMS]  {friendly}: retry "
                    f"{attempt}/{BMS_RETRIES - 1} ..."
                )
                await asyncio.sleep(RETRY_DELAY)

            result = await read_jbd_device(dev, friendly, password=password)

            if result.error is None:
                return result   # success

            err_lower = (result.error or "").lower()
            if not any(t in err_lower for t in _TRANSIENT_ERRORS):
                log.debug(
                    f"  [BMS]  {friendly}: permanent error, "
                    f"not retrying: {result.error}"
                )
                break

        return result   # type: ignore[return-value]  # result is always set

    # Sequential BMS polling
    bms_readings: list[DeviceReading] = []
    for i, entry in enumerate(jbd_pairs):
        if i > 0:
            await asyncio.sleep(INTER_DEVICE_GAP)
        bms_readings.append(await _read_bms_with_retry(entry))

    # Victron reading (scanner still running — fresh nonces available)
    mppt_readings: list[DeviceReading] = []
    for entry in mppt_triples:
        if entry[0] is None:
            _, _, name, ident = entry
            mppt_readings.append(DeviceReading(
                address=ident, name=name, device_type="mppt",
                timestamp=datetime.now().isoformat(timespec="seconds"),
                error="Device not found during scan",
            ))
        else:
            dev, adv_snapshot, name, key = entry
            all_payloads = scanner.victron_payloads(dev.address)
            fresh        = scanner.latest_adv(dev.address)
            adv          = fresh[1] if fresh else adv_snapshot
            mppt_readings.append(
                read_victron_advertisement(dev, adv, name, key, all_payloads)
            )

    # All BLE work complete — release the radio
    await scanner.stop()

    return bms_readings, mppt_readings
