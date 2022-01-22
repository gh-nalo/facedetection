import psutil
import time
import os
import pandas as pd
import platform

NUM = "01"

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


if __name__ == "__main__":

    is_windows = platform.machine() == "AMD64"

    cpu = []
    mem = []
    temperature = []

    try:
        while True:
            v_cpu, v_memory, v_temperature = get_stats()
            cpu.append(v_cpu)
            mem.append(v_memory)
            temperature.append(v_temperature)

            print(v_cpu, v_memory, v_temperature)

    except KeyboardInterrupt:

        zipped = zip(cpu, mem) if is_windows else zip(cpu, mem, temperature)
        columns = ["CPU%", "MEM%"] if is_windows else [
            "CPU%", "MEM%", "CPU_TEMP"]

        df = pd.DataFrame(list(zipped), columns=columns)

        df.to_excel(f"./TimingResults/{NUM}_measurements.xlsx", index=False)
