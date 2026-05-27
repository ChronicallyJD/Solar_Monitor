"""
solar_monitor/jbd.py — JBD / Vatrer BMS GATT protocol
======================================================
Handles everything needed to connect to a JBD-compatible BMS over BLE GATT
and retrieve a "basic info" reading containing pack voltage, current, state
of charge, cell count, and NTC temperatures.

Supported hardware
------------------
- JBD (Jiabaida / Xiaoxiang) BMS modules
- Vatrer Power batteries (use JBD BMS internally, confirmed on SP16S020L16S100A)
- Overkill Solar boards
- Daly BMS (if they expose the ff00/ff01/ff02 GATT profile)

Protocol summary
----------------
Communication is over GATT:
  Service ff00  (or ffe0 / Nordic UART on some firmware)
  TX char ff01  — BMS sends responses; host subscribes with notify
  RX char ff02  — host sends commands; write with or without response

The command to request basic info (register 0x03):
  DD A5 03 00 FF FD 77

Response packet (4-byte header format, confirmed empirically):
  [0]       0xDD  start marker
  [1]       0x03  register echo
  [2]       0x00  status (0x80 = error)
  [3]       N     payload length
  [4..N+3]  payload (big-endian fields, see _parse_jbd_basic_info)
  [N+4-5]   checksum  (0x10000 - sum(reg, len, payload) & 0xFFFF)
  [N+6]     0x77  end marker

Optional password authentication (register 0x06) is sent before the info
request if a password is configured.  Factory default is "0000".
"""

import asyncio
import logging
import struct
from typing import Optional

from bleak import BleakClient
from bleak.backends.device import BLEDevice

from .models import DeviceReading

log = logging.getLogger(__name__)


# ── GATT UUID candidates ──────────────────────────────────────────────────────

# Tried in order; the first with matching TX+RX characteristics wins.
JBD_UUID_CANDIDATES: list[dict] = [
    {   # Standard JBD / Xiaoxiang (most common)
        "service": "0000ff00-0000-1000-8000-00805f9b34fb",
        "tx":      "0000ff01-0000-1000-8000-00805f9b34fb",  # notify (BMS→host)
        "rx":      "0000ff02-0000-1000-8000-00805f9b34fb",  # write  (host→BMS)
    },
    {   # Vatrer / newer JBD firmware  — same char for TX and RX
        "service": "0000ffe0-0000-1000-8000-00805f9b34fb",
        "tx":      "0000ffe1-0000-1000-8000-00805f9b34fb",
        "rx":      "0000ffe1-0000-1000-8000-00805f9b34fb",
    },
    {   # Nordic UART Service (NUS) clone
        "service": "6e400001-b5a3-f393-e0a9-e50e24dcca9e",
        "tx":      "6e400003-b5a3-f393-e0a9-e50e24dcca9e",  # notify
        "rx":      "6e400002-b5a3-f393-e0a9-e50e24dcca9e",  # write
    },
]

# BLE device name substrings that trigger auto-discovery
JBD_NAME_KEYWORDS: tuple[str, ...] = (
    "jbd", "bms", "xiaoxiang", "overkill", "daly", "vatrer", "sp04", "sp16"
)

# Register 0x03 "basic info" request command
BASIC_INFO_CMD = bytes([0xDD, 0xA5, 0x03, 0x00, 0xFF, 0xFD, 0x77])

# Seconds to wait for a notify response before raising TimeoutError
READ_TIMEOUT = 8

# Maximum sane payload length (guards against corrupt length bytes stalling reads)
MAX_PAYLOAD_LEN = 128


# ── Packet helpers ────────────────────────────────────────────────────────────

def _jbd_checksum(payload: bytes) -> bytes:
    """
    Compute the 2-byte JBD checksum.

    Checksum = (0x10000 - sum(payload)) & 0xFFFF, stored big-endian.
    *payload* must include the register and length bytes but not the
    start/end markers.
    """
    chk = (0x10000 - sum(payload)) & 0xFFFF
    return bytes([chk >> 8, chk & 0xFF])


def _packet_complete(buf: bytearray) -> bool:
    """
    Return True when *buf* holds a complete JBD response packet.

    Uses the payload-length field at byte [3] to compute the expected total
    length rather than relying on the 0x77 end marker (which can appear as
    a data byte inside the payload).

    Returns False immediately if the length field exceeds MAX_PAYLOAD_LEN,
    indicating a corrupt packet that will never complete.
    """
    if len(buf) < 4:
        return False
    if buf[0] != 0xDD:
        return False
    n = buf[3]
    if n > MAX_PAYLOAD_LEN:
        return False        # corrupt length — caller should reset buffer
    return len(buf) >= n + 7


def _parse_basic_info(data: bytes) -> dict:
    """
    Parse a JBD/Vatrer BMS "basic info" response (register 0x03).

    Packet layout (4-byte header, big-endian payload):
      [0]       0xDD  start
      [1]       0x03  register echo
      [2]       0x00  status (0x80 = error)
      [3]       N     payload length
      [4..N+3]  payload
      [N+4-5]   checksum
      [N+6]     0x77  end

    Selected payload fields (offset within payload, all big-endian):
      0-1   pack voltage      (10 mV units)
      2-3   pack current      (10 mA, signed; positive = charging)
      19    state of charge   (%)
      21    cell count
      22    NTC count
      23+   NTC temperatures  (0.1 K per LSB; °C = (raw - 2731) / 10)

    Returns:
        dict with keys voltage_v, current_a, power_w, capacity_pct,
        cell_count, temp_c.
    Raises:
        ValueError on any framing, length, or status error.
    """
    if len(data) < 8:
        raise ValueError(f"Response too short ({len(data)} bytes)")
    if data[0] != 0xDD:
        raise ValueError(f"Bad start byte: 0x{data[0]:02X} (expected 0xDD)")
    if data[2] == 0x80:
        raise ValueError(f"BMS returned error code 0x{data[3]:02X}")

    payload_len  = data[3]
    expected_end = payload_len + 6
    if len(data) < expected_end + 1:
        raise ValueError(
            f"Packet truncated: got {len(data)}B, need {expected_end + 1}B  "
            f"raw={data.hex()}"
        )
    if data[expected_end] != 0x77:
        raise ValueError(
            f"Bad end byte at pos {expected_end}: "
            f"0x{data[expected_end]:02X} (expected 0x77)  raw={data.hex()}"
        )

    payload = data[4: 4 + payload_len]
    if len(payload) < 23:
        raise ValueError(f"Payload too short ({len(payload)}B, need 23)")

    voltage_v = struct.unpack_from(">H", payload, 0)[0] * 10 / 1000.0
    current_a = struct.unpack_from(">h", payload, 2)[0] * 10 / 1000.0
    soc        = payload[19]
    cell_count = payload[21]
    ntc_count  = payload[22]

    temps_c = [
        round((struct.unpack_from(">H", payload, 23 + i * 2)[0] - 2731) / 10.0, 1)
        for i in range(ntc_count)
    ]
    return {
        "voltage_v":    round(voltage_v, 3),
        "current_a":    round(current_a, 3),
        "power_w":      round(voltage_v * current_a, 2),
        "capacity_pct": soc,
        "cell_count":   cell_count,
        "temp_c":       temps_c,
    }


# ── GATT service discovery ────────────────────────────────────────────────────

async def _discover_chars(client: BleakClient) -> tuple[str, str]:
    """
    Walk the connected device's GATT table and return (tx_uuid, rx_uuid).

    Tries each entry in JBD_UUID_CANDIDATES in order, then falls back to
    a heuristic search for any service with both a notify and a write
    characteristic.

    Returns:
        (tx_uuid, rx_uuid) — TX notifies, RX accepts writes.
    Raises:
        ValueError if no compatible service is found (GATT table logged at WARNING).
    """
    svcs = client.services

    for svc in svcs:
        for c in svc.characteristics:
            log.debug(f"  GATT  svc={svc.uuid}  char={c.uuid}  props={c.properties}")

    for svc in svcs:
        for candidate in JBD_UUID_CANDIDATES:
            if svc.uuid.lower() != candidate["service"].lower():
                continue
            tx = candidate["tx"].lower()
            rx = candidate["rx"].lower()
            if (any(c.uuid.lower() == tx for c in svc.characteristics) and
                    any(c.uuid.lower() == rx for c in svc.characteristics)):
                log.info(f"    Matched JBD service {svc.uuid}  TX={tx}  RX={rx}")
                return tx, rx

    # Heuristic fallback
    for svc in svcs:
        notifiers = [c for c in svc.characteristics if "notify" in c.properties]
        writers   = [c for c in svc.characteristics
                     if "write" in c.properties or "write-without-response" in c.properties]
        if notifiers and writers:
            tx, rx = notifiers[0].uuid, writers[0].uuid
            log.warning(f"    No known JBD service matched — heuristic TX={tx} RX={rx}")
            return tx, rx

    for svc in svcs:
        for c in svc.characteristics:
            log.warning(f"  GATT  svc={svc.uuid}  char={c.uuid}  props={c.properties}")
    raise ValueError("No compatible JBD service found. See GATT table above.")


# ── GATT reader ───────────────────────────────────────────────────────────────

class JBDGattReader:
    """
    Manages the GATT notify/write exchange with a JBD-compatible BMS.

    Usage::

        async with BleakClient(device) as client:
            raw = await JBDGattReader(client).read_basic_info(password="0000")
            data = _parse_basic_info(raw)

    The reader:
    1. Discovers the correct TX/RX characteristics for this specific device.
    2. Subscribes to TX notifications.
    3. Waits 1 second for the subscription to be fully registered (BlueZ race fix).
    4. Optionally sends the password unlock command (register 0x06).
    5. Sends the basic-info request and reassembles the response from one or
       more notify chunks using length-based framing (not 0x77 sentinel).
    """

    def __init__(self, client: BleakClient) -> None:
        self._client  = client
        self._buf     = bytearray()
        self._event   = asyncio.Event()
        self._tx_uuid: Optional[str] = None
        self._rx_uuid: Optional[str] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _on_notify(self, _sender, data: bytearray) -> None:
        """Notification handler: accumulate bytes and signal when packet complete."""
        self._buf.extend(data)
        # Resync: strip leading bytes that are not the 0xDD start marker.
        # This recovers from leftover bytes from a previous failed read.
        while self._buf and self._buf[0] != 0xDD:
            self._buf.pop(0)
        log.debug(
            f"    notify +{len(data)}B  total={len(self._buf)}B  "
            f"hex={self._buf.hex()}"
        )
        if _packet_complete(self._buf):
            log.debug(f"    packet complete ({len(self._buf)}B)")
            self._event.set()

    async def _write(self, data: bytes) -> None:
        """Write *data* to the RX characteristic, choosing the correct write type."""
        rx_char = self._client.services.get_characteristic(self._rx_uuid)
        if rx_char is None:
            raise ValueError(f"RX characteristic {self._rx_uuid} not found after connect")
        use_response = "write" in rx_char.properties
        log.debug(f"    write {'w/rsp' if use_response else 'w/o rsp'}  {data.hex()}")
        await self._client.write_gatt_char(self._rx_uuid, data, response=use_response)

    async def _send_recv(self, cmd: bytes) -> bytes:
        """Send *cmd* and wait for a complete response packet."""
        self._buf.clear()
        self._event.clear()
        await self._write(cmd)
        await asyncio.wait_for(self._event.wait(), timeout=READ_TIMEOUT)
        return bytes(self._buf)

    # ── Public API ────────────────────────────────────────────────────────────

    async def authenticate(self, password: str) -> None:
        """
        Send the optional password unlock command (register 0x06).

        The BMS replies with a success or error frame.  A rejected password
        raises ValueError immediately rather than letting the subsequent
        basic-info request time out.
        """
        pw_bytes = password.encode("ascii")
        body     = bytes([0x06, len(pw_bytes)]) + pw_bytes
        cmd      = bytes([0xDD, 0x5A]) + body + _jbd_checksum(body) + bytes([0x77])
        reply    = await self._send_recv(cmd)
        if len(reply) >= 3 and reply[2] == 0x80:
            raise ValueError("BMS rejected password — check the password in your config")
        log.debug("    BMS password accepted")

    async def read_basic_info(self, password: Optional[str] = None) -> bytes:
        """
        Read register 0x03 (basic info) from the BMS.

        Performs GATT service discovery, optional authentication, and the
        info request, then returns the raw response bytes for parsing by
        _parse_basic_info().

        Args:
            password: Connection password string (e.g. "0000"), or None.

        Returns:
            Raw response bytes including header and end marker.

        Raises:
            asyncio.TimeoutError: If no complete packet arrives within READ_TIMEOUT.
            ValueError: On authentication failure or GATT service not found.
        """
        self._tx_uuid, self._rx_uuid = await _discover_chars(self._client)
        await self._client.start_notify(self._tx_uuid, self._on_notify)
        # Give BlueZ time to complete the GATT notification subscription on the
        # remote device.  Without this delay the BMS response arrives before the
        # kernel handler is active and the buffer stays empty.
        await asyncio.sleep(1.0)
        try:
            if password:
                await self.authenticate(password)
            return await self._send_recv(BASIC_INFO_CMD)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"No BMS response after {READ_TIMEOUT}s — "
                f"buffer={len(self._buf)}B: {self._buf.hex() or '(empty)'}"
            )
        finally:
            try:
                await self._client.stop_notify(self._tx_uuid)
            except Exception:
                pass


# ── Public reader function ────────────────────────────────────────────────────

async def read_jbd_device(
    device: BLEDevice,
    friendly_name: Optional[str] = None,
    password: Optional[str] = None,
) -> DeviceReading:
    """
    Connect to a JBD BMS, read basic info, and return a DeviceReading.

    Args:
        device:        BLEDevice object from the scanner.
        friendly_name: Dashboard label; falls back to device.name or MAC.
        password:      Optional BMS connection password.

    Returns:
        DeviceReading with voltage, current, power, SoC, cell count, and
        temperatures populated on success, or with error set on failure.
    """
    from datetime import datetime
    ts   = datetime.now().isoformat(timespec="seconds")
    name = friendly_name or device.name or device.address
    r    = DeviceReading(address=device.address, name=name,
                         device_type="bms", timestamp=ts)
    try:
        async with BleakClient(device, timeout=15) as client:
            raw    = await JBDGattReader(client).read_basic_info(password=password)
            parsed = _parse_basic_info(raw)
            for k, v in parsed.items():
                setattr(r, k, v)
        log.info(
            f"  [BMS]  {name}: {r.voltage_v}V  {r.current_a}A  "
            f"{r.power_w}W  SoC={r.capacity_pct}%"
        )
    except Exception as exc:
        r.error = str(exc)
        log.warning(f"  [BMS]  {name}: ERROR -- {exc}")
    return r
