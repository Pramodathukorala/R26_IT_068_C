// ─── How the phone reaches the backends ───────────────────────────────────────
//
// Each component runs as its own service on its own port, and the app talks to
// each one directly:
//
//     Backend/Behaviour detection    Final.ipynb, last cell    :8000
//     Backend/Disease_detection      uvicorn main:app          :8001
//     Backend/Snake&Scropiens…       python server.py          :8002
//
// They are independent processes on purpose - they cannot share one, since
// Disease needs TensorFlow on Python 3.13 in its own .venv, the predator
// backend needs torch on 3.14, and the chicken one is a Jupyter cell.
//
// Two ways to connect, depending on how the device is attached:
//
// 1. USB (works even on a locked-down / AP-isolated Wi-Fi, and when Windows
//    Firewall blocks inbound). Run once per device connection - one per port:
//
//        adb reverse tcp:8000 tcp:8000     # chicken CCTV API
//        adb reverse tcp:8001 tcp:8001     # disease API
//        adb reverse tcp:8002 tcp:8002     # snake & scorpion API
//
//    then flip USE_USB_TUNNEL to true. `adb reverse` is per-connection -
//    re-run it every time the device drops and reconnects.
//
// 2. Wi-Fi (needed for a real, untethered phone). Use the PC's LAN IP.
//    Requirements:
//      - phone and PC on the same network,
//      - every backend bound to 0.0.0.0, not 127.0.0.1,
//      - Windows Firewall allows inbound TCP on each port,
//      - the router does not have AP/client isolation enabled.
//
// NOTE: the PC's LAN IP changes when you switch Wi-Fi networks. Re-check it
// with `ipconfig` (Wireless LAN adapter Wi-Fi → IPv4 Address) rather than
// trusting the value below.
// Verified 2026-08-31: PC = 192.168.8.102 (it was .103 on 08-30 — the PC, phone
// and camera all swap addresses between sessions — re-check with ipconfig
// whenever the app cannot reach a backend).

// Set to `true` when the phone is on USB and `adb reverse` has been run - it is
// the more reliable path when it works, and needs no firewall or router help.
// 2026-09-01: forced on. The router has AP/client isolation enabled - the phone
// (192.168.8.101) and this PC (192.168.8.102) sit on the same /24 but cannot
// ping each other in either direction, so option 2 below cannot work on this
// network at all. Set back to false only on a router that allows client-to-
// client traffic.
const USE_USB_TUNNEL = true;

// This moves constantly - it was .102, then .104, then .102 again across a
// single day (2026-09-01). Re-check `ipconfig` rather than trusting it. Only
// used when USE_USB_TUNNEL is false; the USB tunnel is immune to it.
const LAN_HOST = '192.168.8.102';

export const HOST = USE_USB_TUNNEL ? '127.0.0.1' : LAN_HOST;

// ─── Chicken CCTV backend ─────────────────────────────────────────────────────
// Backend/Behaviour detection (Final.ipynb, last cell). Plain http.server on
// 0.0.0.0:8000 (CHICKEN_API_HOST / CHICKEN_API_PORT override it).
// Heads up: that server walks its port upward if 8000 is busy (8000 -> 8001 ->
// …), which would collide with the disease API. Start it first, or pin it with
// the CHICKEN_API_PORT env var.
export const CHICKEN_API_BASE_URL = `http://${HOST}:8000`;

// ─── Disease detection backend ────────────────────────────────────────────────
// Backend/Disease_detection - FastAPI. Start it with
//     python -m uvicorn main:app --host 0.0.0.0 --port 8001
export const DISEASE_API_BASE_URL = `http://${HOST}:8001`;

// ─── Snake & scorpion (predator) backend ─────────────────────────────────────
// Backend/Snake&Scropiens Detection - `python server.py` (EfficientNet-B3).
// Binds 0.0.0.0:8002 (PREDATOR_API_HOST / PREDATOR_API_PORT override it).
export const PREDATOR_API_BASE_URL = `http://${HOST}:8002`;

// Every backend this app talks to, for the dashboard's health roll-up.
export const SERVICES = [
  { key: 'chicken', label: 'Chicken Behaviour', baseUrl: CHICKEN_API_BASE_URL },
  { key: 'disease', label: 'Disease Detection', baseUrl: DISEASE_API_BASE_URL },
  { key: 'predator', label: 'Predator Alert', baseUrl: PREDATOR_API_BASE_URL },
];
