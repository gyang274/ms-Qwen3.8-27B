#!/usr/bin/env python3
"""Custom temperature -> fan-speed curve for NVIDIA GPUs via NVML (headless, no X needed). Must run as root.

Ramps up immediately, ramps down slowly (hysteresis). Hands control back to the driver on exit.
Fails safe: any NVML read error sets the fan to 100 %.
Health check: warns when the measured fan speed stays well below the commanded speed (worn bearing, stuck fan).

usage: gpu_fan_curve.py [--curve "45:30,55:40,65:55,72:70,78:85,82:100"] [--interval 2] [--down-step 2]
                        [--health-gap 15] [--health-secs 120]
       gpu_fan_curve.py --reset     (restore driver automatic fan control and exit)
"""
import argparse
import signal
import sys
import time

import pynvml as nv

DEFAULT_CURVE = "45:30,55:40,65:55,72:70,78:85,82:100"


def parse_curve(s):
    pts = sorted((int(t), int(f)) for t, f in (p.split(":") for p in s.split(",")))
    assert all(0 <= f <= 100 for _, f in pts), "fan % must be within 0..100"
    return pts


def target_for(temp, pts):
    if temp <= pts[0][0]:
        return pts[0][1]
    for (t0, f0), (t1, f1) in zip(pts, pts[1:]):
        if temp <= t1:
            return round(f0 + (f1 - f0) * (temp - t0) / (t1 - t0))
    return pts[-1][1]


def handles():
    return [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]


def reset_all():
    for h in handles():
        for fan in range(nv.nvmlDeviceGetNumFans(h)):
            nv.nvmlDeviceSetDefaultFanSpeed_v2(h, fan)


def check_health(i, g, args):
    """Warn once per episode when any fan lags its commanded speed by > health_gap for > health_secs."""
    try:
        measured = min(nv.nvmlDeviceGetFanSpeed_v2(g["h"], fan) for fan in range(g["fans"]))
    except nv.NVMLError as e:
        print(f"gpu{i}: WARNING fan speed read failed ({e})", flush=True)
        return
    now = time.monotonic()
    if g["cur"] is not None and measured < g["cur"] - args.health_gap:
        g["lag_since"] = g["lag_since"] or now
        if not g["warned"] and now - g["lag_since"] >= args.health_secs:
            print(f"gpu{i}: WARNING fan health: measured {measured}% vs commanded {g['cur']}% "
                  f"for {now - g['lag_since']:.0f}s (worn bearing or blocked fan?)", flush=True)
            g["warned"] = True
    else:
        if g["warned"]:
            print(f"gpu{i}: fan health recovered: measured {measured}% vs commanded {g['cur']}%", flush=True)
        g["lag_since"], g["warned"] = None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default=DEFAULT_CURVE)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--down-step", type=int, default=2, help="max %% decrease per interval")
    ap.add_argument("--health-gap", type=int, default=15, help="warn if measured fan %% < commanded - gap ...")
    ap.add_argument("--health-secs", type=float, default=120, help="... continuously for this many seconds")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    nv.nvmlInit()
    if args.reset:
        reset_all()
        print("fan control returned to driver")
        return

    pts = parse_curve(args.curve)
    gpus = []
    for h in handles():
        lo, hi = nv.nvmlDeviceGetMinMaxFanSpeed(h)
        gpus.append({"h": h, "fans": nv.nvmlDeviceGetNumFans(h), "min": lo, "max": hi, "cur": None,
                     "lag_since": None, "warned": False})

    def stop(*_):
        reset_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"fan curve {pts}, {len(gpus)} GPU(s)", flush=True)

    try:
        while True:
            for i, g in enumerate(gpus):
                try:
                    temp = nv.nvmlDeviceGetTemperature(g["h"], nv.NVML_TEMPERATURE_GPU)
                    want = target_for(temp, pts)
                except nv.NVMLError as e:
                    temp, want = None, 100
                    print(f"gpu{i}: temperature read failed ({e}), forcing 100%", flush=True)
                if g["cur"] is not None and want < g["cur"]:
                    want = max(want, g["cur"] - args.down_step)
                want = min(max(want, g["min"]), g["max"])
                if want != g["cur"]:
                    for fan in range(g["fans"]):
                        nv.nvmlDeviceSetFanSpeed_v2(g["h"], fan, want)
                    if g["cur"] is None or abs(want - g["cur"]) >= 5:
                        print(f"gpu{i}: {temp}C -> fan {want}%", flush=True)
                    g["cur"] = want
                check_health(i, g, args)
            time.sleep(args.interval)
    finally:
        reset_all()


if __name__ == "__main__":
    main()
