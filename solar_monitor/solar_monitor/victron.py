"""
solar_monitor/victron.py — Victron Instant Readout BLE advertisement parsing
=============================================================================
Decodes Victron Energy device advertisements broadcast over BLE.  No GATT
connection is required — all data arrives in the manufacturer-specific
advertisement payload.

Supported devices
-----------------
- SmartSolar / BlueSolar MPPT charge controllers   (record type 0x01)
- SmartShunt / BMV battery monitors                (record type 0x02, 0x08)
- Phoenix / MultiPlus / Quattro inverters          (record type 0x03, 0x06, 0x0B, 0x0C)
- DC/DC converters (Orion)                         (record type 0x04, 0x09, 0x0D)
- SmartLithium batteries                           (record type 0x05)
- DC Energy Meter / confirming inverter            (record type 0x07)

Advertisement formats
---------------------
Bleak returns ``AdvertisementData.manufacturer_data`` as a dict keyed by
company ID (0x02E1 for Victron).  Bleak **strips the 2-byte company ID**,
so index 0 of the value bytes is the first application-level payload byte.

Two payload formats exist in the wild:

**Format A** — Product Advertisement (outer type 0x10):
  [0]    = 0x10  outer record type
  [1-2]  = model ID (uint16 LE)
  [3]    = readout byte: high nibble = key index, low nibble = record type
  [4]    = counter/flags byte
  [5-6]  = nonce (uint16 LE, increments each beacon)
  [7]    = key-index byte  — NOT key[0]; do not compare against the key
  [8+]   = AES-128-CTR encrypted payload

**Format B** — Extra Manufacturer Data (direct record):
  [0]    = record type
  [1-2]  = nonce (uint16 LE)
  [3]    = key-index byte
  [4+]   = AES-128-CTR encrypted payload

Key identification
------------------
Devices rotate between advertising their own specific record type AND the
generic Solar Charger beacon (0x01).  The scanner accumulates ALL distinct
payloads per MAC.  This module tries them in preference order:

  1. Non-0x01 records with higher type numbers first (more device-specific).
  2. The generic 0x01 Solar Charger beacon last as a fallback.

For each candidate payload the module:
  a. Decrypts using AES-128-CTR with the configured advertisement key.
  b. For record types with a state byte at [0] (solar charger, inverter),
     verifies the state is a known value — random bytes are unlikely to hit
     one of the ~9 valid state codes.
  c. Checks that any decoded voltage is physically plausible (0–150 V).
  d. Uses the first payload that passes all checks.

Encryption key
--------------
The 32-hex-character AES-128 Advertisement key is found in VictronConnect:
  Connect to device > gear icon > Product info
  > scroll to "Instant Readout via Bluetooth" > tap "Show"
  > copy the "Advertisement key"

This is DIFFERENT from the "Encryption key" shown higher on the same screen,
which is used for VE.Smart networking between Victron devices.
"""

import logging
import struct
from datetime import datetime
from typing import Optional

from bleak.backends.device import BLEDevice

from .models import DeviceReading

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

VICTRON_MFR_ID = 0x02E1  # Victron Energy BLE manufacturer ID

# record_type -> (human-readable label, DeviceReading.device_type value)
VICTRON_RECORD_TYPES: dict[int, tuple[str, str]] = {
    0x01: ("Solar Charger",              "mppt"),
    0x02: ("Battery Monitor",            "monitor"),
    0x03: ("Inverter",                   "inverter"),
    0x04: ("DC/DC Converter",            "dcdc"),
    0x05: ("SmartLithium",               "lithium"),
    0x06: ("Inverter RS",                "inverter"),
    0x07: ("DC Energy Meter / Inverter", "inverter"),  # user-confirmed inverter
    0x08: ("SmartShunt IP65",            "monitor"),
    0x09: ("DC-DC Charger",              "dcdc"),
    0x0B: ("Multi RS",                   "inverter"),
    0x0C: ("VE.Bus",                     "inverter"),
    0x0D: ("Orion XS",                   "dcdc"),
}

# BLE device name substrings that trigger auto-discovery
VICTRON_NAME_KEYWORDS: tuple[str, ...] = (
    "victron", "smartsolar", "bluesolar", "bmv", "smartshunt",
    "multiplus", "phoenix", "orion", "multi rs", "quattro",
)

_CHARGER_STATES: dict[int, str] = {
    0: "Off", 2: "Fault", 3: "Bulk", 4: "Absorption",
    5: "Float", 7: "Equalize", 252: "ESS", 255: "Unavailable",
}

_INVERTER_STATES: dict[int, str] = {
    0: "Off", 1: "Low Power", 2: "Fault", 9: "Inverting", 255: "Unavailable",
}

# Record types whose decrypted byte [0] is a charger/inverter state code.
# For other types (BMV, DC meter etc.) byte [0] has a different meaning.
_RECORDS_WITH_STATE: set[int] = {0x01, 0x03, 0x06, 0x07, 0x0B, 0x0C}

# Valid charger/inverter state codes (byte [0] of decrypted payload)
_VALID_STATES: set[int] = {0, 1, 2, 3, 4, 5, 7, 9, 252, 255}


# ── Advertisement extraction ──────────────────────────────────────────────────

def extract_victron_mfr(adv_data) -> Optional[bytes]:
    """
    Return the best usable Victron payload from an AdvertisementData object.

    Victron devices may broadcast:
    - A 4-byte VE.Smart networking beacon (Format A, < 9 bytes) — not Instant Readout
    - A full Instant Readout payload (Format A >= 9 bytes, or Format B >= 5 bytes)

    Bleak may return a single bytes value or a list of bytes values for the same
    manufacturer ID, depending on version.  Both cases are handled.

    Returns:
        The first usable payload bytes, or None if none found.
    """
    mfr = getattr(adv_data, "manufacturer_data", {}) or {}
    raw = mfr.get(VICTRON_MFR_ID)
    if raw is None:
        return None

    payloads = raw if isinstance(raw, list) else [raw]
    for p in payloads:
        if not p:
            continue
        if p[0] == 0x10:
            if len(p) >= 9:
                return p      # Format A Instant Readout
            # else: short VE.Smart beacon — skip
        elif len(p) >= 5:
            return p          # Format B Instant Readout
    return None


def parse_payload(mfr_raw: bytes) -> tuple[int, int, bytes]:
    """
    Extract ``(record_type, nonce_val, ciphertext)`` from a raw Victron payload.

    Handles both Format A (starts with 0x10) and Format B (direct record type).
    See module docstring for the full byte layout of each format.

    Returns:
        ``(record_type, nonce_val, ciphertext)`` on success, or
        ``(0xFF, 0, b"")`` if the payload is too short or unrecognised.
    """
    if mfr_raw[0] == 0x10 and len(mfr_raw) >= 9:
        # Format A: record type in low nibble of byte [3]
        record_type = mfr_raw[3] & 0x0F
        nonce_val   = struct.unpack_from("<H", mfr_raw, 5)[0]
        ciphertext  = mfr_raw[8:]
        return record_type, nonce_val, ciphertext
    elif len(mfr_raw) >= 5:
        # Format B: record type is byte [0]
        record_type = mfr_raw[0]
        nonce_val   = struct.unpack_from("<H", mfr_raw, 1)[0]
        ciphertext  = mfr_raw[4:]
        return record_type, nonce_val, ciphertext
    return 0xFF, 0, b""


def try_decrypt(nonce_val: int, ciphertext: bytes,
                key_bytes: bytes) -> Optional[bytes]:
    """
    AES-128-CTR decrypt a Victron advertisement payload.

    Nonce construction: 2-byte LE counter zero-padded to 16 bytes.  This
    matches the Victron firmware implementation confirmed empirically.

    Args:
        nonce_val:   16-bit advertisement counter.
        ciphertext:  Encrypted payload bytes.
        key_bytes:   16-byte advertisement key.

    Returns:
        Decrypted bytes, or None if the ``cryptography`` package is absent.
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        nonce  = struct.pack("<H", nonce_val) + b"\x00" * 14
        cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(nonce),
                        backend=default_backend())
        dec    = cipher.decryptor()
        return dec.update(ciphertext) + dec.finalize()
    except ImportError:
        return None


# ── Record parsers ────────────────────────────────────────────────────────────

def _parse_solar(dec: bytes) -> dict:
    """
    Parse record type 0x01 — Solar Charger (SmartSolar / BlueSolar MPPT).

    Decrypted payload layout:
      [0]    state        (uint8,  charger state code)
      [1]    error        (uint8,  error code; 0xFF = N/A)
      [2-3]  batt_v       (int16 LE, 0.01 V; 0x7FFF = N/A)
      [4-5]  batt_a       (int16 LE, 0.1 A;  0x7FFF = N/A)
      [6-7]  yield_today  (uint16 LE, 0.01 kWh; 0xFFFF = N/A) -> Wh = val * 10
      [8-9]  pv_power     (uint16 LE, 1 W; 0xFFFF = N/A)
    """
    if len(dec) < 10:
        raise ValueError(f"Solar record too short ({len(dec)}B, need 10)")
    state     = dec[0]
    error     = dec[1]
    batt_mv   = struct.unpack_from("<h", dec, 2)[0]
    batt_ma   = struct.unpack_from("<h", dec, 4)[0]
    yield_raw = struct.unpack_from("<H", dec, 6)[0]
    pv_raw    = struct.unpack_from("<H", dec, 8)[0]

    batt_v   = None if batt_mv  == 0x7FFF else round(batt_mv  * 0.01, 3)
    batt_a   = None if batt_ma  == 0x7FFF else round(batt_ma  * 0.1,  3)
    pv_w     = None if pv_raw   == 0xFFFF else float(pv_raw)
    yield_wh = None if yield_raw == 0xFFFF else round(yield_raw * 10.0, 1)
    batt_w   = (round(batt_v * batt_a, 2)
                if batt_v is not None and batt_a is not None else None)
    return {
        "voltage_v":      batt_v,
        "current_a":      batt_a,
        "power_w":        batt_w,
        "pv_power_w":     pv_w,
        "yield_today_wh": yield_wh,
        "charger_state":  _CHARGER_STATES.get(state, f"0x{state:02X}"),
        "error_code":     error if error != 0xFF else None,
    }


def _parse_inverter(dec: bytes) -> dict:
    """
    Parse record types 0x03 / 0x06 / 0x07 / 0x0B / 0x0C — Inverter variants.

    Layout confirmed empirically from live Victron inverter packets:
      [0]    state        (uint8,  inverter state code)
      [1-2]  alarm_reason (uint16 LE, bitmask)
      [3-4]  batt_v       (uint16 LE, millivolts — NOTE: mV not 0.01V; 0xFFFF = N/A)
      [5-6]  ac_va        (uint16 LE, apparent power W/VA; 0xFFFF = N/A)
      [7-10] bit-packed word (uint32 LE):
               bits  0-14: ac_voltage  (0.01 V; 0x7FFF = N/A)
               bits 15-25: ac_current  (0.1 A;  0x7FF  = N/A)

    Note on battery voltage scaling: the Victron spec document says 0.01 V
    (int16), but live data analysis shows the field contains millivolts
    (uint16) for this hardware.  0xAEFF = 44799 mV = 44.799 V matches
    a slightly-discharged 48 V battery; treating it as int16 * 0.01 gives
    an impossible -207 V.
    """
    if len(dec) < 7:
        raise ValueError(f"Inverter record too short ({len(dec)}B, need 7)")
    state     = dec[0]
    alarm     = struct.unpack_from("<H", dec, 1)[0]
    batt_mV   = struct.unpack_from("<H", dec, 3)[0]   # uint16, millivolts
    ac_va_raw = struct.unpack_from("<H", dec, 5)[0]

    word     = struct.unpack_from("<I", dec, 7)[0] if len(dec) >= 11 else 0
    ac_v_raw = word & 0x7FFF
    ac_i_raw = (word >> 15) & 0x7FF

    batt_v = None if batt_mV  == 0xFFFF else round(batt_mV  * 0.001, 3)
    ac_va  = None if ac_va_raw == 0xFFFF else float(ac_va_raw)
    ac_v   = None if ac_v_raw  == 0x7FFF else round(ac_v_raw * 0.01, 2)
    ac_i   = None if ac_i_raw  == 0x7FF  else round(ac_i_raw * 0.1,  2)

    return {
        "voltage_v":        batt_v,
        "power_w":          ac_va,   # AC apparent power used as the shared power_w field
        "ac_out_power_va":  ac_va,
        "ac_out_voltage_v": ac_v,
        "ac_out_current_a": ac_i,
        "inverter_state":   _INVERTER_STATES.get(state, f"0x{state:02X}"),
        "alarm_reason":     alarm if alarm else None,
    }


def _parse_bmv(dec: bytes) -> dict:
    """
    Parse record type 0x02 — Battery Monitor (SmartShunt, BMV-712/702).

    Layout (all fields are bit-packed per Victron spec):
      [0-1]  ttg          (uint16 LE, minutes; 0xFFFF = N/A)
      [2-3]  batt_v       (int16 LE,  0.01 V; 0x7FFF = N/A)
      [4-5]  alarm        (uint16 LE, bitmask)
      [6-7]  aux          (uint16 LE, context-dependent)
      ...
      bit 96-117: current (22-bit signed, 0.001 A; 0x1FFFFF = N/A)
      bit 140-149: SoC    (10-bit uint, 0.1 %; 0x3FF = N/A)

    Note: byte [0] is NOT a state code for this record type — it is the
    low byte of the time-to-go field.  The state-byte sanity check in
    read_victron_advertisement is intentionally skipped for record 0x02.
    """
    if len(dec) < 16:
        raise ValueError(f"BMV record too short ({len(dec)}B, need 16)")
    ttg_raw = struct.unpack_from("<H", dec, 0)[0]
    batt_mv = struct.unpack_from("<h", dec, 2)[0]
    alarm   = struct.unpack_from("<H", dec, 4)[0]

    # Current: 22-bit signed at bit offset 96 (byte 12, bits 0-21)
    word_i = struct.unpack_from("<I", dec, 12)[0]
    i_raw  = word_i & 0x3FFFFF
    if i_raw & 0x200000:
        i_raw -= 0x400000        # sign-extend from 22 bits
    # 0x1FFFFF = all 21 data bits set = N/A sentinel
    batt_a = None if i_raw == 0x1FFFFF else round(i_raw * 0.001, 3)

    # SoC: 10-bit at bit offset 140 (byte 17 bits 4-13)
    soc_raw = None
    if len(dec) >= 19:
        ws    = struct.unpack_from("<H", dec, 17)[0]
        s     = (ws >> 4) & 0x3FF
        soc_raw = s if s != 0x3FF else None

    batt_v   = None if batt_mv == 0x7FFF else round(batt_mv * 0.01, 3)
    ttg_min  = None if ttg_raw == 0xFFFF else ttg_raw
    soc_pct  = None if soc_raw is None   else round(soc_raw * 0.1, 1)
    batt_w   = (round(batt_v * batt_a, 2)
                if batt_v is not None and batt_a is not None else None)
    return {
        "voltage_v":    batt_v,
        "current_a":    batt_a,
        "power_w":      batt_w,
        "capacity_pct": int(soc_pct) if soc_pct is not None else None,
        "ttg_minutes":  ttg_min,
        "alarm_reason": alarm if alarm else None,
    }


def _parse_dcenergy(dec: bytes) -> dict:
    """
    Parse record type 0x08 — SmartShunt IP65 / DC Energy Meter.

    Similar to BMV but layout differs slightly:
      [0-1]  ttg          (uint16 LE, minutes; 0xFFFF = N/A)
      [2-3]  batt_v       (int16 LE, 0.01 V; 0x7FFF = N/A)
      [4-5]  alarm        (uint16 LE, bitmask)
      ...
      bit 96+: current   (22-bit signed, 0.001 A)
    """
    if len(dec) < 6:
        raise ValueError(f"DC Energy Meter record too short ({len(dec)}B, need 6)")
    ttg_raw = struct.unpack_from("<H", dec, 0)[0]
    batt_mv = struct.unpack_from("<h", dec, 2)[0]
    alarm   = struct.unpack_from("<H", dec, 4)[0]

    batt_v  = None if batt_mv == 0x7FFF else round(batt_mv * 0.01, 3)
    ttg_min = None if ttg_raw == 0xFFFF else ttg_raw

    batt_a = None
    if len(dec) >= 16:
        word_i = struct.unpack_from("<I", dec, 12)[0]
        i_raw  = word_i & 0x3FFFFF
        if i_raw & 0x200000:
            i_raw -= 0x400000
        if i_raw != 0x1FFFFF:
            batt_a = round(i_raw * 0.001, 3)

    batt_w = (round(batt_v * batt_a, 2)
              if batt_v is not None and batt_a is not None else None)
    return {
        "voltage_v":    batt_v,
        "current_a":    batt_a,
        "power_w":      batt_w,
        "ttg_minutes":  ttg_min,
        "alarm_reason": alarm if alarm else None,
    }


# Dispatch table: record_type -> parser function
PARSERS: dict[int, callable] = {
    0x01: _parse_solar,
    0x02: _parse_bmv,
    0x03: _parse_inverter,
    0x04: _parse_bmv,       # DC/DC Converter
    0x05: _parse_bmv,       # SmartLithium
    0x06: _parse_inverter,  # Inverter RS
    0x07: _parse_inverter,  # DC Energy Meter confirmed as inverter
    0x08: _parse_dcenergy,  # SmartShunt IP65
    0x09: _parse_bmv,       # DC-DC Charger
    0x0B: _parse_inverter,  # Multi RS
    0x0C: _parse_inverter,  # VE.Bus
    0x0D: _parse_bmv,       # Orion XS
}


# ── Public reading function ───────────────────────────────────────────────────

def read_victron_advertisement(
    device: BLEDevice,
    adv_data,
    friendly_name: Optional[str],
    enc_key: Optional[str],
    all_payloads: Optional[list[bytes]] = None,
) -> DeviceReading:
    """
    Decode a Victron BLE advertisement and return a DeviceReading.

    Victron devices cycle through advertising multiple record types within a
    single advertisement period.  ``all_payloads`` should contain every
    distinct payload accumulated for this MAC during the scan window so that
    the device-specific record type can be found even if the generic 0x01
    Solar Charger beacon arrived last.

    Selection algorithm:
      1. Merge ``all_payloads`` with any payload present in ``adv_data``.
      2. Sort: device-specific records (non-0x01, higher type number first),
         then the generic 0x01 beacon as a last resort.
      3. For each candidate: decrypt, validate state byte (if applicable),
         validate voltage is in 0–150 V range.
      4. Use the first candidate that passes all checks.

    Args:
        device:       BLEDevice from the scanner.
        adv_data:     AdvertisementData from the scanner callback.
        friendly_name: Dashboard label; falls back to device name or MAC.
        enc_key:       32-hex advertisement key, or None.
        all_payloads:  All distinct payloads accumulated for this MAC.

    Returns:
        DeviceReading with fields populated on success, or with ``error``
        set on failure (decryption error, wrong key, no data, etc.).
    """
    ts   = datetime.now().isoformat(timespec="seconds")
    name = friendly_name or device.name or device.address

    # ── Build candidate list ──────────────────────────────────────────────────
    candidates: list[bytes] = list(all_payloads or [])
    snap = extract_victron_mfr(adv_data)
    if snap and snap not in candidates:
        candidates.append(snap)

    if not candidates:
        log.warning(f"  [Victron] {name}: no advertisement data")
        return DeviceReading(
            address=device.address, name=name,
            device_type="victron", timestamp=ts,
            error="No usable Victron Instant Readout data found in advertisement",
        )

    # ── Sort: prefer the record type most likely to be from THIS device ──────
    # Priority order:
    #   1. The device's own Instant Readout record (NOT the generic 0x01 beacon).
    #      Among non-0x01 types, prefer LOWER numbers — 0x01 devices (MPPT) often
    #      also broadcast 0x02 (BMV) VE.Smart network data; 0x01 < 0x02 so the
    #      MPPT's own data wins over the shared network data.
    #   2. The generic 0x01 Solar Charger beacon last.
    def _priority(p: bytes) -> tuple:
        rt, _, _ = parse_payload(p)
        if rt == 0x01:
            return (2, rt)   # generic beacon — last resort
        return (0, rt)       # device-specific: LOWER type numbers first (0x01 MPPT < 0x02 BMV)

    candidates.sort(key=_priority)

    # Log candidates at INFO — essential for diagnosing key / format issues
    for p in candidates:
        rt, nv, _ = parse_payload(p)
        fmt = "A" if p[0] == 0x10 else "B"
        log.info(
            f"  [Victron] {name}: candidate fmt={fmt} rec=0x{rt:02X} "
            f"nonce=0x{nv:04X} len={len(p)} raw={p.hex()}"
        )

    # ── No key: identify device type only ────────────────────────────────────
    if not enc_key:
        rt, _, _ = parse_payload(candidates[0])
        label, dtype = VICTRON_RECORD_TYPES.get(rt, ("Victron", "victron"))
        log.info(f"  [{label}] {name}: no encryption key configured")
        return DeviceReading(
            address=device.address, name=name,
            device_type=dtype, timestamp=ts,
            charger_state=f"No key ({label})",
            error="Encryption key required for full data",
        )

    # ── Validate key ──────────────────────────────────────────────────────────
    try:
        key_bytes = bytes.fromhex(enc_key[:32])
    except ValueError:
        return DeviceReading(
            address=device.address, name=name,
            device_type="victron", timestamp=ts,
            error="Invalid encryption key: must be exactly 32 hex characters",
        )

    # ── Try each candidate until one decrypts successfully ───────────────────
    last_error = "no candidate payload decrypted successfully"

    for payload in candidates:
        record_type, nonce_val, ciphertext = parse_payload(payload)
        if record_type == 0xFF or not ciphertext:
            continue

        parser = PARSERS.get(record_type)
        if parser is None:
            log.debug(
                f"  [Victron] {name}: no parser for record type "
                f"0x{record_type:02X}, skipping"
            )
            continue

        decrypted = try_decrypt(nonce_val, ciphertext, key_bytes)
        if decrypted is None:
            last_error = (
                "cryptography package not installed — "
                "run: pip install cryptography"
            )
            break   # no point trying further candidates

        # State-byte check: only applicable to charger/inverter record types
        if record_type in _RECORDS_WITH_STATE and decrypted:
            if decrypted[0] not in _VALID_STATES:
                log.debug(
                    f"  [Victron] {name}: rec=0x{record_type:02X} "
                    f"state byte=0x{decrypted[0]:02X} not a valid state, "
                    f"trying next candidate"
                )
                last_error = (
                    f"state byte 0x{decrypted[0]:02X} is not a known "
                    f"charger/inverter state for record type "
                    f"0x{record_type:02X}"
                )
                continue

        # Parse and sanity-check decoded values
        try:
            parsed = parser(decrypted)
        except Exception as exc:
            log.debug(
                f"  [Victron] {name}: rec=0x{record_type:02X} parse "
                f"error ({exc}), trying next candidate"
            )
            last_error = f"parse error for rec=0x{record_type:02X}: {exc}"
            continue

        v_check = parsed.get("voltage_v")
        a_check = parsed.get("current_a")

        # Voltage: 0-150V covers all Victron-compatible battery systems
        if v_check is not None and not (0.0 <= v_check <= 150.0):
            log.debug(
                f"  [Victron] {name}: rec=0x{record_type:02X} "
                f"V={v_check:.2f} outside 0-150V range, "
                f"trying next candidate"
            )
            last_error = (
                f"decoded voltage {v_check:.2f}V is outside the "
                f"physically plausible 0-150V range"
            )
            continue

        # Current: ±2000A catches gross parser mismatches (e.g. BMV parser
        # applied to a Solar Charger record gives hundreds of amps from
        # garbage bit fields)
        if a_check is not None and abs(a_check) > 2000.0:
            log.debug(
                f"  [Victron] {name}: rec=0x{record_type:02X} "
                f"A={a_check:.1f} outside +-2000A range, "
                f"trying next candidate"
            )
            last_error = (
                f"decoded current {a_check:.1f}A is outside the "
                f"physically plausible +-2000A range"
            )
            continue

        # ── Success ───────────────────────────────────────────────────────────
        label, dtype = VICTRON_RECORD_TYPES.get(record_type, ("Victron", "victron"))
        r = DeviceReading(
            address=device.address, name=name,
            device_type=dtype, timestamp=ts,
        )
        for k, val in parsed.items():
            if hasattr(r, k):
                setattr(r, k, val)
        state_str = r.charger_state or r.inverter_state or ""
        log.info(
            f"  [{label}] {name}: "
            f"V={r.voltage_v}  A={r.current_a}  "
            f"pv={r.pv_power_w}W  ac={r.ac_out_power_va}VA  "
            f"state={state_str}"
        )
        return r

    # ── All candidates failed ─────────────────────────────────────────────────
    rt0, _, _ = parse_payload(candidates[0])
    label, dtype = VICTRON_RECORD_TYPES.get(rt0, ("Victron", "victron"))
    error_msg = (
        f"Decryption failed: {last_error}.  "
        f"Make sure you are using the Advertisement key from VictronConnect: "
        f"connect to device > gear icon > Product info > scroll to "
        f"'Instant Readout via Bluetooth' > tap Show > copy Advertisement key."
    )
    log.warning(
        f"  [{label}] {name}: all {len(candidates)} candidate(s) failed "
        f"-- {last_error}"
    )
    return DeviceReading(
        address=device.address, name=name,
        device_type=dtype, timestamp=ts,
        error=error_msg,
    )
