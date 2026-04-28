# set for [1] second window, NOT 2 second. change by searching for "2SECOND"

import argparse
import sys
import time
from pathlib import Path

import serial
from serial import SerialException

HEADER = "time_us,ax,ay,az,gx,gy,gz"
DEFAULT_LAST_TIME_US = 1_996_000 # 2SECOND window
#DEFAULT_LAST_TIME_US = 996_000 # 1SECOND window
PROGRESS_BAR_WIDTH = 30
#OUT_FOLDER = "imu_samples"
OUT_FOLDER = "1sec_imu_samples"

def connect_serial(port: str, baud: int) -> serial.Serial:
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=0.05, write_timeout=1)
            ser.dtr = False
            ser.rts = False
            time.sleep(0.2)
            ser.reset_input_buffer()
            print(f"Connected to {port} at {baud} baud.")
            return ser
        except SerialException:
            print(f"Waiting for {port}...")
            time.sleep(1)


def next_index(prefix: str, folder: Path) -> int:
    idx = 0
    while (folder / f"{prefix}{idx:02d}.txt").exists():
        idx += 1
    return idx


def render_progress(elapsed_s: float, total_s: float) -> None:
    frac = max(0.0, min(1.0, elapsed_s / total_s))
    filled = int(round(frac * PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(f"\r[{bar}] {elapsed_s:4.2f}/{total_s:4.2f} s", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture triggered 2-second IMU gesture windows.")
    parser.add_argument("gesture", help="Label prefix, e.g. up, down, none, circle_cw")
    parser.add_argument("--port", default="COM9")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--folder", default=OUT_FOLDER)
    parser.add_argument("--last-time-us", type=int, default=DEFAULT_LAST_TIME_US)
    parser.add_argument("--arm-command", default="CSV\n", help="Command sent to the ESP32 to start capture") # CSV dump csv output, ARM classifies
    parser.add_argument("--captures", type=int, default=1, help="How many windows to collect before exiting")
    args = parser.parse_args()

    label = args.gesture.strip().lower()
    prefix = f"{label}_"
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)

    ser = connect_serial(args.port, args.baud)
    file_index = next_index(prefix, folder)
    total_capture_s = args.last_time_us / 1_000_000.0

    print("Ready.")
    print("For each sample: steady the device, then press Enter. The script will arm the MCU.")
    print("The progress bar is PC-side only and does not control MCU timing.")

    captures_done = 0
    current_file = None
    current_filename = None

    while captures_done < args.captures:
        try:
            input(f"\n[{captures_done + 1}/{args.captures}] Press Enter to arm capture for '{label}'...")
            print("Enter received")
            print("Sending ARM now...")
            ser.reset_input_buffer()
            ser.write(args.arm_command.encode("utf-8"))
            print("ARM sent")

            started = False
            capture_start_pc = time.monotonic()
            last_progress_update = 0.0

            while True:
                now = time.monotonic()
                if now - last_progress_update >= 0.03:
                    render_progress(now - capture_start_pc, total_capture_s)
                    last_progress_update = now

                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                if line == HEADER:
                    current_filename = folder / f"{prefix}{file_index:02d}.txt"
                    file_index += 1
                    current_file = open(current_filename, "w", encoding="utf-8")
                    current_file.write(line + "\n")
                    current_file.flush()
                    started = True
                    print(f"\nStarted: {current_filename.name}")
                    continue

                if started and current_file is not None:
                    first = line.split(",")[0]
                    if first.isdigit():
                        current_file.write(line + "\n")
                        current_file.flush()
                        t_us = int(first)
                        if t_us >= args.last_time_us:
                            current_file.close()
                            current_file = None
                            captures_done += 1
                            print(f"\nComplete: {current_filename.name}")
                            break

        except KeyboardInterrupt:
            print("\nStopped by user.")
            if current_file is not None:
                current_file.close()
            sys.exit(0)
        except SerialException:
            print("\nSerial connection lost. Reconnecting...")
            try:
                ser.close()
            except Exception:
                pass
            ser = connect_serial(args.port, args.baud)

    ser.close()
    print("All requested captures completed.")


if __name__ == "__main__":
    main()
