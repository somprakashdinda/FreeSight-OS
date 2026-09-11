"""
FreeSight-OS Sleek Terminal Countdown Timer
Usage:
    python countdown.py [duration] [label]
Examples:
    python countdown.py 5m
    python countdown.py 10m "Reviewing GitHub Repository"
    python countdown.py 300s
    python countdown.py 60
"""

import sys
import time
import re

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

def parse_duration(arg: str) -> int:
    """Parses duration string like '5m', '300s', '1h', '45' into seconds."""
    arg = arg.strip().lower()
    m = re.match(r"^(\d+)(s|m|h)?$", arg)
    if not m:
        return 300  # default 5 minutes
    val, unit = int(m.group(1)), m.group(2)
    if unit == 'h':
        return val * 3600
    elif unit == 'm':
        return val * 60
    else:
        return val

def format_time(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def render_bar(fraction: float, length: int = 36) -> str:
    filled = int(round(length * fraction))
    bar = "█" * filled + "░" * (length - filled)
    return bar

def run_countdown(total_seconds: int, label: str = "FreeSight-OS Countdown"):
    print()
    print("=" * 64)
    print(f"  ⏳ {label.upper()}")
    print(f"  Target Duration: {format_time(total_seconds)} ({total_seconds}s)")
    print("=" * 64)
    print()

    start_time = time.time()
    try:
        while True:
            elapsed = int(time.time() - start_time)
            remaining = max(0, total_seconds - elapsed)
            fraction = min(1.0, elapsed / max(1, total_seconds))
            pct = fraction * 100.0

            bar = render_bar(fraction)
            time_str = format_time(remaining)

            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  |  Remaining: {time_str}  ")
            sys.stdout.flush()

            if remaining <= 0:
                break
            time.sleep(0.5)

        print("\n")
        print("=" * 64)
        print(f"  🎉 TIME'S UP! - {label}")
        print("=" * 64)
        print()

        # Audio alert
        if HAS_WINSOUND:
            for freq in (880, 1174, 1318, 1760):
                winsound.Beep(freq, 120)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  [!] Countdown interrupted by user.\n")

if __name__ == "__main__":
    duration_arg = sys.argv[1] if len(sys.argv) > 1 else "5m"
    label_arg = sys.argv[2] if len(sys.argv) > 2 else "FreeSight-OS Timer"
    total_secs = parse_duration(duration_arg)
    run_countdown(total_secs, label_arg)
