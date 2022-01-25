import psutil
import os
import pandas as pd
import platform
from FilePaths import Base
import argparse


def get_raspberry_cpu_temp() -> float:
    result = 0.0

    if os.path.isfile('/sys/class/thermal/thermal_zone0/temp'):
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            line = f.readline().strip()

        if line.isdigit():
            result = float(line) / 1000

    return result


def get_stats() -> list:
    cpu = psutil.cpu_percent(1.0)
    mem = psutil.virtual_memory().percent
    cpu_temp = get_raspberry_cpu_temp()

    return cpu, mem, cpu_temp


def save_results(cpu, mem, temp, name) -> None:
    is_windows = platform.machine() == "AMD64"

    zipped = zip(cpu, mem) if is_windows else zip(cpu, mem, temp)
    columns = ["CPU%", "MEM%"] if is_windows else ["CPU%", "MEM%", "CPU_TEMP"]

    df = pd.DataFrame(list(zipped), columns=columns)

    df.to_excel(os.path.join(Base.RESULTS, name) + "_measurements.xlsx", index=False)


if __name__ == "__main__":
    cpu = []
    mem = []
    temp = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm To Run Inference On"
    )

    args = vars(parser.parse_args())
    name = args["algorithm"]

    try:
        while True:

            v_cpu, v_memory, v_temperature = get_stats()

            cpu.append(v_cpu)
            mem.append(v_memory)
            temp.append(v_temperature)

            print(v_cpu, v_memory, v_temperature)

            # Stop measurements when face detection results are saved
            if os.path.exists(os.path.join(Base.RESULTS, name) + ".xlsx"):
                save_results(cpu, mem, temp, name)
                break

    # Stop measurements when interrupting script
    except KeyboardInterrupt:

        save_results(cpu, mem, temp, name)
