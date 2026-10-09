"""Step 9: คืนข้อมูลเดิมผ่านระบบเต็ม และแสดงกราฟ 6 แถวในหน้าเดียว.

ใน Jupyter ใช้ %run ตามด้วย path ของไฟล์นี้ แล้วดูตัวแปร result ต่อได้
แสดงกราฟด้วย plt.show() เท่านั้น ไม่มีการบันทึกรูปหรือข้อมูลอัตโนมัติ
"""

import contextlib
import io
from pathlib import Path
import runpy

import numpy as np

from step02_scppm_encoder import encode_scppm
from step04_plot_ppm_output import (
    make_ppm_slots, atmospheric_turbulence_noise, poisson_noise,
)
from step06_scppm_decoder import decode_scppm
from step08_frame_decoder import decode_first_part


# ปรับค่าทดลองตรงนี้: Noise ทั้งสองเปิด/ปิดแยกกันได้
USE_TURBULENCE = True
USE_POISSON = True
SIGNAL_PHOTONS = 10.0

# ระดับสัญญาณและจำนวนรอบสำหรับ BER curve
BER_PHOTON_LEVELS = [0.5, 1, 2, 3, 5, 7, 10]
BER_TRIALS_PER_LEVEL = 10

# ตัดเฉพาะช่วงที่แสดงกราฟ แต่เข้ารหัสและถอดรหัสเต็มทุก block เสมอ
NUMBER_OF_BITS = 128
NUMBER_OF_SYMBOLS = 64
PLOT_BLOCK = 0  # 0 คือ block แรก


def run_full_system(
    use_turbulence=True, use_poisson=True, signal_photons=10.0, trial_number=0
):
    """รันทุก block แล้วคืนข้อมูลต้นฉบับ สัญญาณแต่ละขั้น และข้อมูลที่ถอดได้."""
    if not np.isfinite(signal_photons) or signal_photons < 0:
        raise ValueError("signal_photons ต้องเป็นค่าจำกัดและไม่น้อยกว่า 0")
    if not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("trial_number ต้องเป็นจำนวนเต็มไม่น้อยกว่า 0")

    # ใช้ Step 1 เดิม และซ่อน print ที่ท้ายไฟล์ขณะโหลด
    folder = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        first_part = runpy.run_path(str(folder / "step01_input_preparation.py"))

    original_data = np.asarray(first_part["data_bit"], dtype=np.uint8)
    input_blocks = first_part["data"]
    frame_bits = first_part["Nbits"]  # ความยาว frame ที่ทั้งสองฝั่งตกลงกัน
    decoded_blocks = []
    channel_blocks = []

    for block_index, input_bits in enumerate(input_blocks):
        symbols = encode_scppm(input_bits)
        x_t = make_ppm_slots(symbols).astype(float) * signal_photons

        # Atmospheric turbulence: เปลี่ยนความแรงของ pulse
        np.random.seed(1001 + block_index + trial_number * 100)
        n_atmospheric = np.zeros_like(x_t)
        if use_turbulence:
            n_atmospheric = atmospheric_turbulence_noise(x_t)
        x_after_atmosphere = x_t + n_atmospheric

        # Poisson: สุ่มจำนวนโฟตอนจากสัญญาณหลังผ่านบรรยากาศ
        np.random.seed(2001 + block_index + trial_number * 100)
        n_poisson = np.zeros_like(x_t)
        if use_poisson:
            n_poisson = poisson_noise(x_after_atmosphere)

        # ตัว decoder รับเฉพาะสัญญาณนี้ ไม่ใช้ต้นฉบับหรือ noise มาช่วยตัดสิน
        r_t = x_t + n_atmospheric + n_poisson
        decoded_blocks.append(decode_scppm(r_t))
        channel_blocks.append({
            "x_t": x_t,
            "n_atmospheric": n_atmospheric,
            "x_after_atmosphere": x_after_atmosphere,
            "n_poisson": n_poisson,
            "r_t": r_t,
        })

    # ส่งผลจาก Viterbi เข้า decoder ส่วน frame ตามไฟล์ที่ผู้ใช้ให้มา
    # ผลตรวจยังใช้ภายในฟังก์ชันเดิม แต่ไม่เพิ่มรายงานแยกบนจอหรือกราฟ
    frames, _ = decode_first_part(decoded_blocks, frame_bits)
    if frames:
        recovered_data = np.concatenate([frame[0] for frame in frames])
    else:
        recovered_data = np.array([], dtype=np.uint8)

    # BER ใช้ข้อมูล payload เต็มทั้ง frame ไม่ใช่เฉพาะช่วงที่ตัดมาแสดงกราฟ
    if len(recovered_data) == len(original_data):
        ber = float(np.mean(recovered_data != original_data))
    else:
        ber = None

    return {
        "original_data": original_data,
        "recovered_data": recovered_data,
        "ber": ber,
        "decoded_blocks": decoded_blocks,
        "channel_blocks": channel_blocks,
        "use_turbulence": use_turbulence,
        "use_poisson": use_poisson,
        "signal_photons": signal_photons,
    }


def plot_full_system(
    result, number_of_bits=128, number_of_symbols=64, block_index=0, show=True
):
    """แสดง 6 กราฟ; บิตใช้ payload index ส่วนสัญญาณใช้ slot index."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if number_of_bits < 1 or number_of_symbols < 1:
        raise ValueError("จำนวนบิตและ symbols ที่แสดงต้องไม่น้อยกว่า 1")
    if not 0 <= block_index < len(result["channel_blocks"]):
        raise ValueError("block_index อยู่นอกช่วง blocks ที่มี")

    original = result["original_data"][:number_of_bits]
    recovered = result["recovered_data"][:number_of_bits]
    channel = result["channel_blocks"][block_index]
    number_of_slots = min(number_of_symbols * 4, len(channel["x_t"]))
    slot_edges = np.arange(number_of_slots + 1)
    slot_centers = np.arange(number_of_slots) + 0.5
    bit_edges = np.arange(len(original) + 1)

    figure, axes = plt.subplots(6, 1, figsize=(15, 14), constrained_layout=True)
    figure.set_facecolor("#f5f7fb")
    figure.suptitle(
        "SCPPM: from original data to recovered data\n"
        f"Turbulence: {'ON' if result['use_turbulence'] else 'OFF'}   |   "
        f"Poisson: {'ON' if result['use_poisson'] else 'OFF'}   |   "
        f"Mean signal photons/pulse: {result['signal_photons']:g}",
        fontsize=15, fontweight="bold",
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    # แถว 1: ข้อมูลเดิมก่อนเข้ากระบวนการฝั่งส่ง
    axes[0].stairs(original, bit_edges, color="#2463a6", linewidth=1.7)
    axes[0].set_title("1. Original data - before encoding", loc="left")

    # แถว 2: สัญญาณที่ห่อ frame และเข้ารหัสเป็น PPM แล้ว ก่อนผ่าน noise
    axes[1].stairs(channel["x_t"][:number_of_slots], slot_edges,
                   color="#2463a6", linewidth=1.3)
    axes[1].set_title("2. Transmitted x(t) - encoded 4-PPM before noise", loc="left")
    axes[1].set_ylabel("Mean photons / slot")
    axes[1].set_ylim(bottom=0)
    axes[1].margins(y=0.15)

    # แถว 3: แสดงผล turbulence เทียบกับสัญญาณส่งที่ใช้จริง
    axes[2].stairs(channel["x_t"][:number_of_slots], slot_edges,
                   color="#8390a2", linewidth=1.1, label="x(t): before atmosphere")
    axes[2].stairs(channel["x_after_atmosphere"][:number_of_slots], slot_edges,
                   color="#169c85", linewidth=1.2, label="x(t) + n_atmospheric")
    axes[2].set_title("3. After atmospheric turbulence", loc="left")
    axes[2].set_ylabel("Mean photons / slot")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_ylim(bottom=0)
    axes[2].margins(y=0.3)

    # แถว 4: ซ้อนสัญญาณก่อนและหลัง Poisson ให้ดูการเปลี่ยนแปลงตรงกันทุก slot
    axes[3].stairs(channel["x_after_atmosphere"][:number_of_slots], slot_edges,
                   color="#8390a2", linewidth=1.1, label="Before Poisson")
    axes[3].stairs(channel["r_t"][:number_of_slots], slot_edges,
                   color="#b87724", linewidth=1.2, label="After Poisson: r(t)")
    axes[3].set_title("4. Before and after Poisson photon counting", loc="left")
    axes[3].set_ylabel("Photons / slot")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(loc="upper right", fontsize=9)

    # แถว 5: เมื่อเปิด Poisson สัญญาณรับคือจำนวนโฟตอนที่ตรวจพบ
    received = channel["r_t"][:number_of_slots]
    if result["use_poisson"]:
        axes[4].vlines(slot_centers, 0, received, color="#8a55b5", linewidth=1.3)
        axes[4].scatter(slot_centers, received, color="#8a55b5", s=7)
        axes[4].set_ylabel("Photon counts / slot")
        axes[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[4].stairs(received, slot_edges, color="#8a55b5", linewidth=1.3)
        axes[4].set_ylabel("Mean photons / slot")
    axes[4].set_ylim(bottom=0)
    axes[4].set_title("5. Received signal r(t) - input to the SCPPM decoder", loc="left")

    # สัญญาณสี่แถวใช้ตำแหน่ง slot เดียวกัน แต่ไม่ได้ตรงกับ payload bit แบบ 1:1
    for axis in axes[1:5]:
        axis.set_xlim(0, number_of_slots)
        axis.set_xlabel(f"Slot index - Block {block_index + 1}")

    # แถว 6: เปรียบเทียบข้อมูลหลังถอดครบทุกขั้นกับข้อมูลต้นฉบับ
    axes[5].stairs(original, bit_edges, color="#2463a6", linewidth=2.2,
                   alpha=0.55, label="Original")
    if len(recovered):
        axes[5].stairs(recovered, np.arange(len(recovered) + 1),
                       color="#d64d51", linewidth=1.3, linestyle="--", label="Recovered")
    else:
        axes[5].text(0.5, 0.5, "Frame decoder returned no frame",
                     transform=axes[5].transAxes, ha="center")
    axes[5].set_title("6. Recovered data - after all decoding stages", loc="left")
    axes[5].legend(loc="upper right", fontsize=9)

    for axis in (axes[0], axes[5]):
        axis.set_xlim(0, len(original))
        axis.set_ylim(-0.15, 1.45)
        axis.set_yticks([0, 1])
        axis.set_ylabel("Bit")
        axis.set_xlabel("Payload bit index")

    # show=False ใช้ตอนสร้างหลายหน้าต่างก่อนสั่ง plt.show() พร้อมกัน
    if show:
        plt.show()
    return figure


def calculate_ber_curve(
    photon_levels, trials_per_level=10, use_turbulence=True, use_poisson=True
):
    """ทดลองหลายระดับ photons/pulse แล้วรวม errors ก่อนคำนวณ BER."""
    photon_levels = np.asarray(photon_levels, dtype=float)
    if photon_levels.ndim != 1 or len(photon_levels) == 0:
        raise ValueError("photon_levels ต้องเป็นรายการหนึ่งมิติที่ไม่ว่าง")
    if np.any(~np.isfinite(photon_levels)) or np.any(photon_levels < 0):
        raise ValueError("photon_levels ต้องเป็นค่าจำกัดและไม่น้อยกว่า 0")
    if not isinstance(trials_per_level, int) or trials_per_level < 1:
        raise ValueError("trials_per_level ต้องเป็นจำนวนเต็มอย่างน้อย 1")

    measured_ber = []
    plotted_ber = []
    zero_error = []

    for photons in photon_levels:
        total_errors = 0
        total_bits = 0

        for trial_number in range(trials_per_level):
            trial = run_full_system(
                use_turbulence=use_turbulence,
                use_poisson=use_poisson,
                signal_photons=float(photons),
                trial_number=trial_number,
            )
            original = trial["original_data"]
            recovered = trial["recovered_data"]
            if len(recovered) != len(original):
                raise RuntimeError("คำนวณ BER ไม่ได้ เพราะความยาว recovered payload ไม่ตรง")

            total_errors += int(np.count_nonzero(recovered != original))
            total_bits += len(original)

        ber = total_errors / total_bits
        measured_ber.append(ber)
        zero_error.append(total_errors == 0)

        # ค่า 0 วางบนแกน log ไม่ได้ จึงวาง marker ที่ 1/จำนวนบิตเป็นขีดจำกัดบน
        if total_errors == 0:
            plotted_ber.append(1 / total_bits)
        else:
            plotted_ber.append(ber)

        print(
            f"photons/pulse={photons:g}: BER={ber:.6g} "
            f"จาก {trials_per_level} trials"
        )

    return {
        "photon_levels": photon_levels,
        "measured_ber": np.asarray(measured_ber),
        "plotted_ber": np.asarray(plotted_ber),
        "zero_error": np.asarray(zero_error),
        "trials_per_level": trials_per_level,
        "use_turbulence": use_turbulence,
        "use_poisson": use_poisson,
    }


def plot_ber_curve(ber_curve, show=True):
    """แสดง Payload BER เทียบ mean signal photons/pulse บนแกน log."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    figure.set_facecolor("#f5f7fb")
    photon_levels = ber_curve["photon_levels"]
    plotted_ber = ber_curve["plotted_ber"]
    zero_error = ber_curve["zero_error"]

    axis.semilogy(
        photon_levels, plotted_ber, color="#d64d51", marker="o",
        linewidth=1.8, markersize=6, label="Measured payload BER",
    )

    # สามเหลี่ยมหัวลงบอกว่าจุดนั้นไม่พบ error และค่าจริงต่ำกว่าตำแหน่ง marker
    if np.any(zero_error):
        axis.scatter(
            photon_levels[zero_error], plotted_ber[zero_error],
            color="#2463a6", marker="v", s=70,
            label="0 errors: upper-limit marker",
        )

    axis.set_xlabel("Mean signal photons per pulse")
    axis.set_ylabel("Payload BER after decoding")
    axis.set_xticks(photon_levels)
    axis.grid(True, which="both", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9)

    axis.set_title(
        "SCPPM payload BER curve\n"
        f"Turbulence: {'ON' if ber_curve['use_turbulence'] else 'OFF'}   |   "
        f"Poisson: {'ON' if ber_curve['use_poisson'] else 'OFF'}   |   "
        f"{ber_curve['trials_per_level']} trials per point",
        fontweight="bold",
    )
    if show:
        plt.show()
    return figure


if __name__ == "__main__":
    result = run_full_system(USE_TURBULENCE, USE_POISSON, SIGNAL_PHOTONS)
    figure = plot_full_system(
        result, NUMBER_OF_BITS, NUMBER_OF_SYMBOLS, PLOT_BLOCK, show=False
    )
    ber_curve = calculate_ber_curve(
        BER_PHOTON_LEVELS,
        BER_TRIALS_PER_LEVEL,
        USE_TURBULENCE,
        USE_POISSON,
    )
    ber_figure = plot_ber_curve(ber_curve, show=False)

    # เปิดกราฟระบบและ BER curve เป็นคนละหน้าต่าง
    import matplotlib.pyplot as plt
    plt.show()
