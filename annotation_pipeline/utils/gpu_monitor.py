from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt


class GPUMonitor:

    def __init__(self, output_dir: Path, interval: int = 1):

        self.output_dir = output_dir
        self.interval = interval
        self.running = False
        self.data = []

    def _worker(self):

        start = time.time()

        while self.running:

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            try:

                gpu, memory, power = map(
                    float,
                    result.stdout.strip().split(", ")
                )

                self.data.append(
                    (
                        time.time() - start,
                        gpu,
                        memory,
                        power,
                    )
                )

            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        self.thread.join()

    def save(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_file = self.output_dir / "gpu_usage.csv"

        with open(
            csv_file,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "time",
                    "gpu",
                    "memory",
                    "power",
                ]
            )

            writer.writerows(self.data)

        # ------------------------------------------------------------
        # Extract arrays
        # ------------------------------------------------------------

        t = [x[0] for x in self.data]
        gpu = [x[1] for x in self.data]
        mem = [x[2] / 1024 for x in self.data]   # MB -> GB
        power = [x[3] for x in self.data]

        if not t:
            return

        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------

        duration = t[-1]

        avg_gpu = sum(gpu) / len(gpu)
        peak_gpu = max(gpu)

        avg_power = sum(power) / len(power)
        peak_power = max(power)

        avg_mem = sum(mem) / len(mem)
        peak_mem = max(mem)

        # ============================================================
        # FIGURE 1 : Benchmark Plot
        # ============================================================

        fig, ax = plt.subplots(
            3,
            1,
            figsize=(15, 9),
            sharex=True,
        )

        fig.suptitle(
            "Gemma-3-12B GPU Benchmark",
            fontsize=18,
            fontweight="bold",
        )

        fig.text(
            0.5,
            0.955,
            (
                f"Duration: {duration:.1f} s    "
                f"Avg GPU: {avg_gpu:.1f}%    "
                f"Peak GPU: {peak_gpu:.0f}%    "
                f"Avg Power: {avg_power:.1f} W    "
                f"Peak Power: {peak_power:.1f} W    "
                f"Avg Memory: {avg_mem:.2f} GB"
            ),
            ha="center",
        )

        # GPU

        ax[0].plot(t, gpu)

        ax[0].set_title("GPU Utilization")

        ax[0].set_ylabel("GPU Util (%)")

        ax[0].grid(True)

        # Power

        ax[1].plot(t, power)

        ax[1].scatter(
            t,
            power,
            s=10,
            label="Power Spike",
        )

        ax[1].legend()

        ax[1].set_title("GPU Power Draw")

        ax[1].set_ylabel("Power (W)")

        ax[1].grid(True)

        # Memory

        ax[2].plot(t, mem)

        ax[2].set_title("GPU Memory Usage")

        ax[2].set_ylabel("Memory (GB)")

        ax[2].set_xlabel("Time (s)")

        ax[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(
            self.output_dir / "gpu_benchmark.png",
            dpi=300,
        )

        plt.close(fig)

        # ============================================================
        # FIGURE 2 : Combined Plot
        # ============================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            t,
            gpu,
            label="GPU Util (%)",
        )

        plt.plot(
            t,
            mem,
            label="Memory (GB)",
        )

        plt.plot(
            t,
            power,
            label="Power (W)",
        )

        plt.xlabel("Time (s)")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "gpu_usage.png",
            dpi=300,
        )

        plt.close()