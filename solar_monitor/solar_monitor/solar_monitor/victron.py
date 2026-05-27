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

    Format A layout (per Victron Extra Manufacturer Data spec):
      [0]    0x10  outer PDU type
      [1-2]  uint16 LE  model ID (Victron product ID)
      [3]    high nibble = key index, low nibble = record type
      [4]    counter byte
      [5-6]  uint16 LE  IV / nonce
      [7]    first byte of encryption key (for verification)
      [8+]   AES-128-CTR ciphertext

    Format B layout:
      [0]    record type
      [1-2]  uint16 LE  IV / nonce
      [3]    first byte of encryption key
      [4+]   ciphertext

    Returns:
        ``(record_type, nonce_val, ciphertext)`` on success, or
        ``(0xFF, 0, b"")`` if the payload is too short or unrecognised.
    """
    if mfr_raw[0] == 0x10 and len(mfr_raw) >= 9:
        model_id    = struct.unpack_from("<H", mfr_raw, 1)[0]
        record_type = mfr_raw[3] & 0x0F
        nonce_val   = struct.unpack_from("<H", mfr_raw, 5)[0]
        ciphertext  = mfr_raw[8:]
        log.debug(
            f"    Format A: model=0x{model_id:04X}({model_id}) "
            f"rec=0x{record_type:02X} nonce=0x{nonce_val:04X}"
        )
        return record_type, nonce_val, ciphertext
    elif len(mfr_raw) >= 5:
        record_type = mfr_raw[0]
        nonce_val   = struct.unpack_from("<H", mfr_raw, 1)[0]
        ciphertext  = mfr_raw[4:]
        log.debug(
            f"    Format B: rec=0x{record_type:02X} nonce=0x{nonce_val:04X}"
        )
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

    Layout per official Victron Extra Manufacturer Data specification.
    All bit offsets are relative to decrypted[0] (spec "Start bit 32").

    Bit   Bytes    Bits  Scale      NA value  Field
    ────  ───────  ────  ─────────  ────────  ──────────────────────────────
      0   [0]        8   —          0xFF      device_state
      8   [1]        8   —          0xFF      charger_error
     16   [2-3]     16   0.01 V     0x7FFF    battery_voltage  (int16 signed)
     32   [4-5]     16   0.1 A      0x7FFF    battery_current  (int16 signed)
     48   [6-7]     16   0.01 kWh   0xFFFF    yield_today      (uint16)
     64   [8-9]     16   1 W        0xFFFF    pv_power         (uint16)
     80   [10] b0    9   0.1 A      0x1FF     load_current     (uint9 packed)
     89           (39)              —         unused

    yield_today: multiply raw value by 10 to get Wh (0.01 kWh × 1000 ÷ 10 = 10 Wh).

    load_current: present on MPPT models with a load output (12V/24V units).
    Stored in bits 0-8 of byte [10].  May be zero or NA on units without a
    load output; callers should treat zero as "no load" not "missing field".
    """
    if len(dec) < 10:
        raise ValueError(f"Solar Charger record too short ({len(dec)}B, need 10)")

    state     = dec[0]
    error     = dec[1]
    batt_mv   = struct.unpack_from("<h", dec, 2)[0]   # int16, 0.01V
    batt_ma   = struct.unpack_from("<h", dec, 4)[0]   # int16, 0.1A
    yield_raw = struct.unpack_from("<H", dec, 6)[0]   # uint16, 0.01kWh
    pv_raw    = struct.unpack_from("<H", dec, 8)[0]   # uint16, 1W

    # load_current: 9-bit uint at bit 80 (byte 10 bits 0-8)
    load_ma = None
    if len(dec) >= 12:
        word10  = struct.unpack_from("<H", dec, 10)[0]
        lc_raw  = word10 & 0x1FF       # 9 bits
        load_ma = None if lc_raw == 0x1FF else round(lc_raw * 0.1, 1)

    batt_v   = None if batt_mv  == 0x7FFF else round(batt_mv  * 0.01, 3)
    batt_a   = None if batt_ma  == 0x7FFF else round(batt_ma  * 0.1,  3)
    pv_w     = None if pv_raw   == 0xFFFF else float(pv_raw)
    yield_wh = None if yield_raw == 0xFFFF else round(yield_raw * 10.0, 1)
    batt_w   = round(batt_v * batt_a, 2) if (batt_v is not None and batt_a is not None) else None

    return {
        "voltage_v":      batt_v,
        "current_a":      batt_a,
        "power_w":        batt_w,
        "pv_power_w":     pv_w,
        "yield_today_wh": yield_wh,
        "load_current_a": load_ma,
        "charger_state":  _CHARGER_STATES.get(state, f"0x{state:02X}"),
        "error_code":     error if error != 0xFF else None,
    }


def _parse_inverter(dec: bytes) -> dict:
    """
    Parse record type 0x03 — Inverter (Phoenix, Quattro, MultiPlus classic).

    Layout per official Victron Extra Manufacturer Data specification
    (https://wiki.victronenergy.com/rend/ble/extra_manufacturer_data).
    All bit offsets are relative to the start of the decrypted payload
    (spec "Start bit 32" = our decrypted[0]).

    Field            Spec bit  Bytes    Bits  Scale    NA value
    ──────────────── ────────  ───────  ────  ───────  ────────
    device_state       32       [0]       8   —        0xFF
    alarm_reason       40       [1-2]    16   —        —
    battery_voltage    56       [3-4]    16   0.01V    0x7FFF  (int16 signed)
    ac_apparent_power  72       [5-6]    16   1 VA     0xFFFF  (uint16)
    ac_voltage         88       [7] b0   15   0.01V    0x7FFF  (bit-packed)
    ac_current        103       [7] b15  11   0.1A     0x7FF   (bit-packed)

    Note on battery_voltage: the spec defines this as int16 * 0.01V, but
    live data from a 48V unit produces -207V with that interpretation.
    The same bytes read as uint16 * 0.001V (millivolts) give 44.8V which
    is physically correct.  This hardware appears to use unsigned millivolt
    units rather than the signed 0.01V units described in the spec.

    Note on ac_apparent_power: THIS is the correct output power field.
    It must be read directly from bytes [5-6]; do NOT compute it from V*I.
    The V*I product is unreliable because previous attempts to find the
    right current scale (0.1A, 0.4A) both produced wrong results at
    different load levels, indicating the bit-packed current field may not
    represent AC output current for this hardware variant.
    """
    if len(dec) < 7:
        raise ValueError(f"Inverter record too short ({len(dec)}B, need 7)")

    state     = dec[0]
    alarm     = struct.unpack_from("<H", dec, 1)[0]   # uint16 alarm bitmask
    batt_cv   = struct.unpack_from("<h", dec, 3)[0]   # int16 LE, 0.01V (per spec)
    ac_va_raw = struct.unpack_from("<H", dec, 5)[0]   # uint16 LE, 1VA (per spec)

    # AC voltage (15 bits) and AC current (11 bits) bit-packed into bytes [7:11]
    word     = struct.unpack_from("<I", dec, 7)[0] if len(dec) >= 11 else 0
    ac_v_raw = word & 0x7FFF          # bits 0-14, 0.01V (per spec)
    ac_i_raw = (word >> 15) & 0x7FF   # bits 15-25, 0.1A  (per spec)

    log.debug(
        f"  [inverter 0x03 fields] dec={dec.hex()}  "
        f"state=0x{state:02X}  alarm=0x{alarm:04X}  "
        f"batt_cv={batt_cv} ({batt_cv*0.01:.2f}V)  "
        f"ac_va_raw={ac_va_raw}  "
        f"ac_v_raw={ac_v_raw} ({ac_v_raw*0.01:.2f}V)  "
        f"ac_i_raw={ac_i_raw} ({ac_i_raw*0.1:.1f}A)"
    )

    batt_v = None if batt_cv   == 0x7FFF else round(batt_cv   * 0.01, 2)
    ac_va  = None if ac_va_raw == 0xFFFF else int(ac_va_raw)
    ac_v   = None if ac_v_raw  == 0x7FFF else round(ac_v_raw  * 0.01, 2)
    ac_i   = None if ac_i_raw  == 0x7FF  else round(ac_i_raw  * 0.1,  2)

    return {
        "voltage_v":        batt_v,
        "power_w":          ac_va,         # AC apparent power — the dedicated spec field
        "ac_out_power_va":  ac_va,
        "ac_out_voltage_v": ac_v,
        "ac_out_current_a": ac_i,
        "inverter_state":   _INVERTER_STATES.get(state, f"0x{state:02X}"),
        "alarm_reason":     alarm if alarm else None,
    }


def _parse_inverter_0x07(dec: bytes) -> dict:
    """
    Parse record type 0x07 — VE.Bus Smart Dongle (custom dongle layout).

    Confirmed device
    ----------------
    The 48V-2400W device in this installation is a **Victron VE.Bus Smart
    Dongle** attached to a MultiPlus-II 48/5000/70-95 120V. The dongle
    broadcasts a custom non-bit-packed payload that does NOT match either:
      - the official spec for 0x03 (Inverter), nor
      - the published 0x0C (VE.Bus) layout from earlier dongle firmwares.

    Field layout (empirically determined from 10 sequential live packets)
    -------------------------------------------------------------------

    Byte    Type        Status        Field
    ──────  ─────────   ────────────  ───────────────────────────────────────
    [0]     uint8       CONFIRMED     device_state (0x09=Inverting, 0xFF=NA)
    [1:3]   uint16 LE   CONFIRMED     battery_voltage in millivolts
    [3]     0xFF        MARKER        constant — not a data field
    [4]     uint8       VARYING       slow-changing field; likely battery
                                      temperature with +40°C offset, but
                                      not yet confirmed against ground truth
    [5:7]   0x0096      MARKER        constant 150 — not instantaneous power
    [7]     0x00        MARKER        constant
    [8]     uint8       VARYING       fast-changing field; clearly AC output
                                      load indicator but scale unknown
    [9:13]  0xFFBE_0800 MARKER        constant end sentinel

    Across the 10 captured packets while inverter was running varying load:
        bytes[1:3] (batt mV):   51712 to 52480 mV (51.71-52.48V) ✓ matches 48V bank
        byte[4]:                46, 47, 49  (3 unique values, slow change)
        byte[8]:                16, 32, 56, 88, 96, 104, 112, 120, 160
                                (9 unique, all multiples of 8, fast change)

    Why ac_out_power and ac_out_current are reported as None
    --------------------------------------------------------
    The scale of byte[8] cannot be determined without a packet captured at
    the same instant as a known VictronConnect AC Out Power reading.
    Plausible scales:
        byte[8] *  2 W:  raw 153 → 306W (matches one VC reading)
        byte[8] *  4 W:  raw  76 → 304W
        byte[8] * 0.05A: raw  64 → 3.2A (matches one VC current reading of 3.19A)
        byte[8] /  8 :    raw 32  → 4A AC current
    All are consistent with the limited data we have.

    To calibrate
    ------------
    1. Run the monitor with `--log-level DEBUG`.
    2. At the same instant, note VC's AC Out Power and AC Out Current.
    3. Read the `[inverter 0x07]` log line — note the `byte[8]` value.
    4. Compute scale = VC_power / byte_8_value.
    5. Set ``_SCALE_WATTS`` below (e.g. to 2.0 if scale comes out 2.0 W/unit)
       or ``_SCALE_AMPS`` (e.g. 0.05 if 0.05A/unit gives correct current).

    Reference
    ---------
    - keshavdv/victron-ble — has no parser for this dongle variant
    - Fabian-Schmidt/esphome-victron_ble — lists VE.Bus as record 0x0C
      with rich bit-packed fields; this dongle broadcasts 0x07 instead
      with a stripped-down byte-aligned layout
    - Victron spec (2022-12-14) — labelled 0x07 "TBD"; never updated
    """
    if len(dec) < 5:
        raise ValueError(f"Inverter 0x07 record too short ({len(dec)}B, need 5)")

    state     = dec[0]
    batt_mV   = struct.unpack_from("<H", dec, 1)[0]
    # bytes[3:5] as uint16 LE consistently reads in the 120-128V range (raw 12031-
    # 12799 at 0.01V scale). byte[3] is always 0xFF which is suspicious — it
    # could be a sentinel/marker rather than the low byte of a 16-bit field.
    # But the values match VC's 120V reading, so we include it with a caveat.
    ac_v_raw  = struct.unpack_from("<H", dec, 3)[0]

    batt_v = None if batt_mV  in (0xFFFF, 0x7FFF) else round(batt_mV  * 0.001, 3)
    ac_v   = None if ac_v_raw in (0xFFFF, 0x7FFF) else round(ac_v_raw * 0.01,  2)

    # byte[4]: slow-changing field, likely battery temperature
    raw_byte4 = dec[4] if len(dec) >= 5 else None
    # byte[8]: AC power indicator, fast-changing
    raw_load  = dec[8] if len(dec) >= 9 else None

    # Calibration constants — set these once you have a paired VC reading
    _SCALE_WATTS = None   # e.g. 2.0 if byte[8] is AC power in 2W units
    _SCALE_AMPS  = None   # e.g. 0.05 if byte[8] is AC current in 0.05A units

    ac_w = (raw_load * _SCALE_WATTS) if (raw_load is not None and _SCALE_WATTS) else None
    ac_a = (raw_load * _SCALE_AMPS)  if (raw_load is not None and _SCALE_AMPS)  else None

    # Speculative battery temperature: byte[4] - 40 (Victron's typical offset)
    # Only report if it's in a plausible range
    batt_temp = None
    if raw_byte4 is not None and 30 <= raw_byte4 <= 100:
        batt_temp = raw_byte4 - 40   # speculative

    log.info(
        f"  [inverter 0x07] dec={dec.hex()}  "
        f"state=0x{state:02X}  "
        f"batt_mV={batt_mV}  "
        f"byte[4]={raw_byte4}  byte[8]={raw_load}  "
        f"(power@2W: {raw_load*2 if raw_load else '?'}W, "
        f"power@4W: {raw_load*4 if raw_load else '?'}W)"
    )

    return {
        "voltage_v":          batt_v,
        "power_w":            ac_w,
        "ac_out_power_va":    ac_w,
        "ac_out_voltage_v":   ac_v,
        "ac_out_current_a":   ac_a,
        "raw_load_indicator": raw_load,   # byte[8] — varies with AC load
        "inverter_state":     _INVERTER_STATES.get(state, f"0x{state:02X}"),
        "alarm_reason":       None,
    }


def _parse_inverter_rs(dec: bytes) -> dict:
    """
    Parse record type 0x06 — Inverter RS.

    Different layout from the plain Inverter (0x03) — includes PV power,
    yield today, and a real AC output power field.

    Field             Spec bit  Bytes    Bits  Scale    NA value
    ───────────────── ────────  ───────  ────  ───────  ────────
    device_state        32       [0]       8   —        0xFF
    charger_error       40       [1]       8   —        0xFF
    battery_voltage     48       [2-3]    16   0.01V    0x7FFF  (int16)
    battery_current     64       [4-5]    16   0.1A     0x7FFF  (int16)
    pv_power            80       [6-7]    16   1W       0xFFFF  (uint16)
    yield_today         96       [8-9]    16   0.01kWh  0xFFFF  (uint16)
    ac_out_power       112       [10-11]  16   1W       0x7FFF  (int16 signed)
    """
    if len(dec) < 12:
        raise ValueError(f"Inverter RS record too short ({len(dec)}B, need 12)")

    state     = dec[0]
    error     = dec[1]
    batt_mv   = struct.unpack_from("<h", dec, 2)[0]   # int16, 0.01V
    batt_ma   = struct.unpack_from("<h", dec, 4)[0]   # int16, 0.1A
    pv_raw    = struct.unpack_from("<H", dec, 6)[0]   # uint16, 1W
    yield_raw = struct.unpack_from("<H", dec, 8)[0]   # uint16, 0.01kWh
    ac_pwr    = struct.unpack_from("<h", dec, 10)[0]  # int16, 1W

    batt_v   = None if batt_mv  == 0x7FFF else round(batt_mv  * 0.01,  3)
    batt_a   = None if batt_ma  == 0x7FFF else round(batt_ma  * 0.1,   3)
    pv_w     = None if pv_raw   == 0xFFFF else float(pv_raw)
    yield_wh = None if yield_raw == 0xFFFF else round(yield_raw * 10.0, 1)
    ac_va    = None if ac_pwr   == 0x7FFF else float(ac_pwr)
    batt_w   = round(batt_v * batt_a, 2) if (batt_v and batt_a is not None) else None

    return {
        "voltage_v":        batt_v,
        "current_a":        batt_a,
        "power_w":          ac_va,
        "pv_power_w":       pv_w,
        "yield_today_wh":   yield_wh,
        "ac_out_power_va":  ac_va,
        "charger_state":    _CHARGER_STATES.get(state, f"0x{state:02X}"),
        "inverter_state":   _INVERTER_STATES.get(state, f"0x{state:02X}"),
        "error_code":       error if error != 0xFF else None,
    }



def _parse_bmv(dec: bytes) -> dict:
    """
    Parse record type 0x02 — Battery Monitor (SmartShunt, BMV-712/702).

    Layout per official Victron spec (bit offsets relative to decrypted[0]):

    Bit   Bytes     Bits  Scale     NA value  Field
    ───── ───────   ────  ────────  ────────  ─────────────────
      0   [0-1]      16   1 min     0xFFFF    TTG (time to go)
     16   [2-3]      16   0.01V     0x7FFF    battery voltage (int16)
     32   [4-5]      16   —         —         alarm reason
     48   [6-7]      16   varies    —         aux (voltage/temp/mid)
     64   [8]         2   —         0x3       aux input mode
     66   [8.25]     22   0.001A    0x3FFFFF  battery current (int22)
     88   [11]       20   0.1Ah     0xFFFFF   consumed Ah (not parsed)
    108   [13.5]     10   0.1%      0x3FF     SOC

    IMPORTANT — aux_input vs battery_current alignment:
    The 2-bit aux_input field at bit 64 (byte 8 bits 0-1) precedes the
    22-bit battery_current field at bit 66 (byte 8 bits 2-23).
    Reading a uint32 at byte 8 and masking bits 0-21 is WRONG — it
    captures 2 aux_input bits + only 20 current bits.
    Correct: aux_input = word & 0x3; i_raw = (word >> 2) & 0x3FFFFF.

    Note: byte [0] is the LOW BYTE OF TTG, not a state code.  The
    state-byte sanity check is intentionally not applied to this record.
    """
    if len(dec) < 16:
        raise ValueError(f"BMV record too short ({len(dec)}B, need 16)")
    ttg_raw = struct.unpack_from("<H", dec, 0)[0]
    batt_mv = struct.unpack_from("<h", dec, 2)[0]
    alarm   = struct.unpack_from("<H", dec, 4)[0]

    # aux_input (2 bits) then battery_current (22 bits) share byte 8
    word_i    = struct.unpack_from("<I", dec, 8)[0]
    aux_input = word_i & 0x3                    # bits 0-1: aux mode
    i_raw_u22 = (word_i >> 2) & 0x3FFFFF        # bits 2-23: 22-bit raw

    # NA sentinel for the 22-bit current field is 0x3FFFFF (all bits set).
    # This check MUST happen against the raw unsigned value BEFORE sign
    # extension. After sign extension 0x3FFFFF becomes -1, which is also a
    # valid current value, so the post-sign-extension comparison is wrong.
    if i_raw_u22 == 0x3FFFFF:
        i_signed = None
    else:
        i_signed = i_raw_u22 - 0x400000 if (i_raw_u22 & 0x200000) else i_raw_u22

    # SoC: 10 bits at bit 108 = byte 13 bit 4
    soc_raw = None
    if len(dec) >= 15:
        ws      = struct.unpack_from("<H", dec, 13)[0]
        s       = (ws >> 4) & 0x3FF
        soc_raw = s if s != 0x3FF else None

    batt_v   = None if batt_mv == 0x7FFF   else round(batt_mv * 0.01, 3)
    ttg_min  = None if ttg_raw == 0xFFFF   else ttg_raw
    batt_a   = None if i_signed is None    else round(i_signed * 0.001, 3)
    soc_pct  = None if soc_raw is None     else round(soc_raw * 0.1,   1)
    batt_w   = round(batt_v * batt_a, 2) if (batt_v is not None and batt_a is not None) else None

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
        word_i    = struct.unpack_from("<I", dec, 12)[0]
        i_raw_u22 = word_i & 0x3FFFFF
        # NA sentinel 0x3FFFFF must be checked BEFORE sign extension
        if i_raw_u22 != 0x3FFFFF:
            i_signed = i_raw_u22 - 0x400000 if (i_raw_u22 & 0x200000) else i_raw_u22
            batt_a   = round(i_signed * 0.001, 3)

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
# Mapping based on:
#  - Official Victron "Extra Manufacturer Data" spec (2022-12-14 PDF)
#  - Fabian-Schmidt/esphome-victron_ble (widely-tested community implementation)
#  - keshavdv/victron-ble (reference Python library)
#
# Per the 2022 spec the record types are:
#   0x01 Solar Charger     0x05 SmartLithium       0x09 Smart Battery Protect
#   0x02 Battery Monitor   0x06 Inverter RS        0x0A Lynx Smart BMS
#   0x03 Inverter          0x07 GX-Device (TBD)    0x0B Multi RS
#   0x04 DC/DC Converter   0x08 AC Charger         0x0C VE.Bus
#                                                  0x0D DC Energy Meter
#                                                  0x0E Orion XS
#
# Note on 0x07: Fabian-Schmidt's ESPHome component lists 0x07 as "AC Charger"
# in current firmware (the 2022 spec called it "GX-Device TBD"). However, the
# 48V/2400W device in this installation broadcasts 0x07 with a layout that
# matches NEITHER the 0x03 Inverter spec NOR the 0x08 spec. The parser
# _parse_inverter_0x07 extracts the three confirmed fields (state, battery V,
# AC out V) and leaves power/current as None pending live cross-reference.
PARSERS: dict[int, callable] = {
    0x01: _parse_solar,           # Solar Charger (MPPT)
    0x02: _parse_bmv,             # Battery Monitor (SmartShunt, BMV-712/702)
    0x03: _parse_inverter,        # Inverter (Phoenix) — classic spec layout
    0x04: _parse_bmv,             # DC/DC Converter (BMV-like layout)
    0x05: _parse_bmv,             # SmartLithium (BMV-like layout)
    0x06: _parse_inverter_rs,     # Inverter RS
    0x07: _parse_inverter_0x07,   # Newer firmware variant — empirical layout
    0x08: _parse_dcenergy,        # AC Charger / SmartShunt IP65
    0x09: _parse_bmv,             # Smart Battery Protect
    0x0B: _parse_inverter_rs,     # Multi RS (same layout as Inverter RS)
    0x0C: _parse_inverter,        # VE.Bus (same as Inverter)
    0x0D: _parse_dcenergy,        # DC Energy Meter
    0x0E: _parse_bmv,             # Orion XS (BMV-like layout)
}


# ── Public reading function ───────────────────────────────────────────────────

def read_victron_advertisement(
    device: BLEDevice,
    adv_data,
    friendly_name: Optional[str],
    enc_key: Optional[str],
    all_payloads: Optional[list[bytes]] = None,
    device_type_override: Optional[str] = None,
) -> DeviceReading:
    """
    Decode a Victron BLE advertisement and return a DeviceReading.

    Victron devices cycle through advertising multiple record types within a
    single advertisement period.  ``all_payloads`` should contain every
    distinct payload accumulated for this MAC during the scan window so that
    the device-specific record type can be found even if the generic 0x01
    Solar Charger beacon arrived last.

    Args:
        device:               BLEDevice from the scanner.
        adv_data:             AdvertisementData from the scanner callback.
        friendly_name:        Dashboard label; falls back to device name or MAC.
        enc_key:              32-hex advertisement key, or None.
        all_payloads:         All distinct payloads accumulated for this MAC.
        device_type_override: Explicit device type from config (e.g. "inverter",
                              "mppt").  When set, payloads whose decoded
                              device_type does not match are skipped, and the
                              final reading always carries this type regardless
                              of which record type was decoded.

    Returns:
        DeviceReading with fields populated on success, or with ``error``
        set on failure.
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

    # Log candidates at DEBUG — use --log-level DEBUG to see them when
    # diagnosing key/format issues. These can total hundreds of lines per
    # poll cycle when many Victron devices are present, so they're not
    # shown at the default INFO level.
    if log.isEnabledFor(logging.DEBUG):
        for p in candidates:
            rt, nv, _ = parse_payload(p)
            fmt = "A" if p[0] == 0x10 else "B"
            log.debug(
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

    # Build a set of acceptable record types when a type override is declared.
    # This prevents e.g. a BMV record being used for a configured inverter.
    _TYPE_TO_RECORDS: dict[str, set[int]] = {
        "mppt":     {0x01},
        "monitor":  {0x02, 0x08},
        "inverter": {0x03, 0x06, 0x07, 0x0B, 0x0C},
        "dcdc":     {0x04, 0x09, 0x0D},
        "lithium":  {0x05},
        "meter":    {0x07},
    }
    allowed_records: Optional[set[int]] = None
    if device_type_override:
        allowed_records = _TYPE_TO_RECORDS.get(device_type_override)
        if allowed_records is None:
            log.warning(
                f"  [Victron] {name}: unknown type override "
                f"'{device_type_override}' — ignoring"
            )

    # ── Try each candidate until one decrypts successfully ───────────────────
    last_error = "no candidate payload decrypted successfully"

    for payload in candidates:
        record_type, nonce_val, ciphertext = parse_payload(payload)
        if record_type == 0xFF or not ciphertext:
            log.debug(f"  [Victron] {name}: skipping unparseable payload {payload.hex()}")
            continue

        # Skip records that don't match the declared device type
        if allowed_records is not None and record_type not in allowed_records:
            log.debug(
                f"  [Victron] {name}: rec=0x{record_type:02X} skipped "
                f"(config type={device_type_override}, allowed={sorted(allowed_records)})"
            )
            continue

        parser = PARSERS.get(record_type)
        if parser is None:
            log.debug(
                f"  [Victron] {name}: no parser for rec=0x{record_type:02X}, skipping"
            )
            continue

        log.debug(
            f"  [Victron] {name}: trying rec=0x{record_type:02X} "
            f"nonce=0x{nonce_val:04X} cipher={ciphertext.hex()}"
        )

        decrypted = try_decrypt(nonce_val, ciphertext, key_bytes)
        if decrypted is None:
            last_error = (
                "cryptography package not installed — "
                "run: pip install cryptography"
            )
            break   # no point trying further candidates

        log.debug(f"  [Victron] {name}: decrypted={decrypted.hex()}")

        # State-byte check: only applicable to charger/inverter record types
        if record_type in _RECORDS_WITH_STATE and decrypted:
            if decrypted[0] not in _VALID_STATES:
                log.debug(
                    f"  [Victron] {name}: rec=0x{record_type:02X} "
                    f"state=0x{decrypted[0]:02X} invalid, trying next"
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
        # Config-declared type overrides the record-type inferred type
        if device_type_override:
            dtype = device_type_override
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
