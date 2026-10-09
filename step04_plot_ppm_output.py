"""Step 4: แสดงกราฟ 4-PPM ของทุก block ในหน้าต่างเดียว."""

import contextlib
import io
from pathlib import Path
import runpy
import sys

import numpy as np

from step02_scppm_encoder import encode_scppm


# ทำให้ข้อความภาษาไทยแสดงได้บน Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_ppm_slots(symbols):
    """แปลงหมายเลข 4-PPM symbols เป็น pulse slots แบบ one-hot."""
    slots = np.zeros(len(symbols) * 4, dtype=np.uint8)

    for symbol_number, symbol in enumerate(symbols):
        # 4-PPM หนึ่ง symbol มี 4 slots และมี pulse เพียงหนึ่ง slot
        pulse_position = symbol_number * 4 + int(symbol)
        slots[pulse_position] = 1

    return slots


def atmospheric_turbulence_noise(x_t):
    """สร้าง atmospheric turbulence แล้วคืนเฉพาะ noise n(t)."""
    sigma = 0.20

    # h(t) เป็น log-normal fading และมีค่าเฉลี่ยประมาณ 1
    h_t = np.random.lognormal(
        mean=-(sigma ** 2) / 2,
        sigma=sigma,
        size=np.size(x_t),
    )

    # n_atmospheric = h(t)x(t) - x(t)
    n_atmospheric = x_t * (h_t - 1)
    return n_atmospheric


def poisson_noise(x_t):
    """สร้าง photon-counting Poisson noise แล้วคืนเฉพาะ noise n(t)."""
    background_photons = 0.10

    # ค่าเฉลี่ย Poisson มาจาก signal photons และ background photons
    poisson_mean = np.clip(x_t, 0, None) + background_photons
    photon_counts = np.random.poisson(poisson_mean)

    # ทำให้ x(t) + n_poisson เท่ากับจำนวนโฟตอนที่ตรวจได้
    n_poisson = photon_counts - x_t
    return n_poisson


def load_ppm_blocks():
    """อ่านข้อมูลจาก Step 1 แล้วส่งผ่าน SCPPM encoder ใน Step 2."""
    folder = Path(__file__).resolve().parent
    first_part_path = folder / "step01_input_preparation.py"

    # ซ่อน print(data) ซึ่งอยู่ท้ายไฟล์ต้นฉบับ
    with contextlib.redirect_stdout(io.StringIO()):
        first_part = runpy.run_path(str(first_part_path))

    input_blocks = first_part["data"]
    ppm_blocks = []

    for block in input_blocks:
        ppm_blocks.append(encode_scppm(block))

    return ppm_blocks


def plot_all_blocks(ppm_blocks):
    """แสดงกราฟ symbol และ pulse ของทั้งสาม blocks ในหน้าต่างเดียว."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ไม่พบ Matplotlib: ติดตั้งด้วย 'python -m pip install matplotlib'"
        ) from error

    # แสดง 16 symbols แรกของทุก block เพื่อให้กราฟอ่านง่าย
    number_of_symbols = 80

    # หนึ่ง block ใช้หนึ่งแถว: ซ้ายเป็นค่า symbol และขวาเป็น pulse slots
    figure, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle("4-PPM Output: Blocks 1-3", fontsize=16)

    for block_index in range(3):
        symbols = ppm_blocks[block_index][:number_of_symbols]
        symbol_indexes = np.arange(number_of_symbols)
        slots = make_ppm_slots(symbols)
        slot_edges = np.arange(len(slots) + 1)

        # x(t) คือ PPM pulse slots ก่อนผ่าน noise
        x_t = slots.astype(float)

        # Atmospheric turbulence=================================================
        n_atmospheric = np.zeros_like(x_t)
        # ใส่ # หน้าบรรทัดต่อไปนี้เพื่อปิด Atmospheric turbulence
        n_atmospheric = atmospheric_turbulence_noise(x_t)
        x_after_atmosphere = x_t + n_atmospheric

        # Poisson photon-counting noise=========================================
        n_poisson = np.zeros_like(x_t)
        # ใส่ # หน้าบรรทัดต่อไปนี้เพื่อปิด Poisson noise
        # n_poisson = poisson_noise(x_after_atmosphere)

        # Received signal: r(t) = x(t) + n(t)
        n_t = n_atmospheric + n_poisson
        r_t = x_t + n_atmospheric + n_poisson

        symbol_graph = axes[block_index, 0]
        pulse_graph = axes[block_index, 1]

        # กราฟซ้าย: ค่า symbol ที่ออกจาก PPM mapper
        symbol_graph.step(symbol_indexes, symbols, where="mid", color="tab:blue")
        symbol_graph.scatter(symbol_indexes, symbols, color="tab:blue", s=24)
        symbol_graph.set_title(f"Block {block_index + 1}: PPM symbols")
        symbol_graph.set_xlabel("Symbol index")
        symbol_graph.set_ylabel("Value")
        symbol_graph.set_yticks([0, 1, 2, 3])
        symbol_graph.set_ylim(-0.3, 3.3)
        symbol_graph.grid(alpha=0.3)

        # กราฟขวา: เปรียบเทียบ x(t), n(t) และ r(t)
        pulse_graph.stairs(x_t, slot_edges, color="tab:blue", label="x(t)")
        pulse_graph.stairs(n_t, slot_edges, color="tab:orange", label="n(t)")
        pulse_graph.stairs(r_t, slot_edges, color="tab:red", label="r(t) = x(t) + n(t)")
        pulse_graph.set_title(f"Block {block_index + 1}: received PPM signal")
        pulse_graph.set_xlabel("Slot index")
        pulse_graph.set_ylabel("Amplitude")
        pulse_graph.grid(alpha=0.3)
        pulse_graph.legend(fontsize=8)

        # เส้นประแบ่งหนึ่ง symbol ต่อ 4 slots
        for boundary in slot_edges[::4]:
            pulse_graph.axvline(
                boundary,
                color="gray",
                linestyle="--",
                linewidth=0.6,
            )

    # เปิดหน้าต่างกราฟอย่างเดียว ไม่มีการ save รูป
    plt.show()


def main():
    ppm_blocks = load_ppm_blocks()
    print("กำลังแสดงกราฟ Block 1, Block 2 และ Block 3 ในหน้าต่างเดียว")
    plot_all_blocks(ppm_blocks)


if __name__ == "__main__":
    main()
