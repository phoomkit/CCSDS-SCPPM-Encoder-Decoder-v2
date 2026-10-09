"""Step 7: ตรวจ decoder และทดลอง channel สี่กรณีด้วยข้อมูลเต็ม block."""

import contextlib
import io
from pathlib import Path
import runpy

import numpy as np

from step02_scppm_encoder import (
    convolutional_encode, code_interleave, accumulate, ppm_symbol_map, encode_scppm,
)
from step04_plot_ppm_output import (
    make_ppm_slots, atmospheric_turbulence_noise, poisson_noise,
)
from step06_scppm_decoder import (
    ppm_hard_demapper, inverse_accumulator, code_deinterleave,
    viterbi_decode, decode_scppm,
)


def check_decoder():
    """ตรวจย้อนแต่ละ stage และการแก้บิตผิดหนึ่งตำแหน่งของ Viterbi."""
    rng = np.random.default_rng(123)
    input_bits = rng.integers(0, 2, 5040, dtype=np.uint8)
    input_bits[-2:] = 0

    coded = convolutional_encode(input_bits)
    interleaved = code_interleave(coded)
    accumulated = accumulate(interleaved)
    symbols = ppm_symbol_map(accumulated)
    slots = make_ppm_slots(symbols)

    np.testing.assert_array_equal(ppm_hard_demapper(slots), accumulated)
    np.testing.assert_array_equal(inverse_accumulator(accumulated), interleaved)
    np.testing.assert_array_equal(code_deinterleave(interleaved), coded)
    np.testing.assert_array_equal(viterbi_decode(coded), input_bits)
    np.testing.assert_array_equal(decode_scppm(slots), input_bits)

    # พิสูจน์ว่า Viterbi แก้ error ได้ในกรณีทดสอบนี้ ไม่ใช่เพียงคืนบิตเดิม
    damaged = coded.copy()
    damaged[300] = damaged[300] ^ 1
    np.testing.assert_array_equal(viterbi_decode(damaged), input_bits)

    # บิตสุดท้ายของ accumulator ไม่จำเป็นต้องเป็น 0
    odd_bits = np.zeros(15120, dtype=np.uint8)
    odd_bits[0] = 1
    np.testing.assert_array_equal(inverse_accumulator(accumulate(odd_bits)), odd_bits)
    assert not np.any(inverse_accumulator(np.zeros(15120, dtype=np.uint8)))

    try:
        decode_scppm(slots[:320])
    except ValueError:
        pass
    else:
        raise AssertionError("Decoder must reject a plot-only prefix")

    print("PASS: inverse stages, full-block round trip, single-bit Viterbi correction")


def load_input_blocks():
    """โหลดข้อมูลจริงจาก Step 1 โดยไม่แก้ไฟล์ต้นฉบับ."""
    folder = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        first_part = runpy.run_path(str(folder / "step01_input_preparation.py"))
    return first_part["data"]


def run_channel_example(use_turbulence=True, use_poisson=True, signal_photons=1.0):
    """ทดลองเต็มทุก block; เปิด/ปิดสอง noise แยกกันด้วย True/False.

    signal_photons คือค่าเฉลี่ยโฟตอนต่อ pulse ก่อน fading ไม่ใช่ amplitude ของบิต
    คืน input_blocks กับ decoded_blocks เพื่อดูผลต่อใน Jupyter ได้
    """
    if not np.isfinite(signal_photons) or signal_photons < 0:
        raise ValueError("signal_photons ต้องเป็นค่าจำกัดและไม่น้อยกว่า 0")
    input_blocks = load_input_blocks()
    decoded_blocks = []
    total_errors = 0
    block_errors = 0

    print(f"\nTurbulence={use_turbulence}, Poisson={use_poisson}, photons/pulse={signal_photons}")
    print("Block | tied PPM symbols | coded errors / 15120 | decoded errors / 5038")

    for block_number, input_bits in enumerate(input_blocks, start=1):
        ppm_symbols = encode_scppm(input_bits)
        # ใช้ครบ 7560 symbols = 30240 slots ไม่ใช่ prefix ที่ตัดไว้ plot
        x_t = make_ppm_slots(ppm_symbols).astype(float) * signal_photons

        # กำหนด seed แยกตาม block เพื่อทำซ้ำการทดลองได้
        np.random.seed(1000 + block_number)
        n_atmospheric = np.zeros_like(x_t)
        if use_turbulence:
            n_atmospheric = atmospheric_turbulence_noise(x_t)
        x_after_atmosphere = x_t + n_atmospheric

        np.random.seed(2000 + block_number)
        n_poisson = np.zeros_like(x_t)
        if use_poisson:
            n_poisson = poisson_noise(x_after_atmosphere)

        r_t = x_t + n_atmospheric + n_poisson

        # Receiver ใช้เฉพาะ r_t: ไม่ลบ noise จริงหรือดูบิตต้นฉบับเพื่อตัดสิน
        received_accumulated = ppm_hard_demapper(r_t)
        received_interleaved = inverse_accumulator(received_accumulated)
        received_coded = code_deinterleave(received_interleaved)
        decoded = viterbi_decode(received_coded)
        decoded_blocks.append(decoded)

        # ส่วนประเมินผลเท่านั้นที่รู้บิตต้นฉบับ
        expected_coded = convolutional_encode(input_bits)
        coded_errors = int(np.count_nonzero(received_coded != expected_coded))
        # ตัด tail 2 บิตออกจากตัวหาร BER เพราะ Viterbi บังคับ end state 00
        errors = int(np.count_nonzero(decoded[:-2] != input_bits[:-2]))
        total_errors += errors
        if errors > 0:
            block_errors += 1

        # รายงานจำนวน symbols ที่ไม่มีผู้ชนะเพียง slot เดียว
        four_slots = r_t.reshape(-1, 4)
        maxima = four_slots.max(axis=1, keepdims=True)
        tied_symbols = int(np.count_nonzero((four_slots == maxima).sum(axis=1) > 1))
        print(f"{block_number:5d} | {tied_symbols:16d} | {coded_errors:20d} | {errors:21d}")

        # เมื่อปิดทั้งสอง noise และมี pulse จริง ต้องตรงครบ 5040 บิต
        if not use_turbulence and not use_poisson and signal_photons > 0:
            np.testing.assert_array_equal(decoded, input_bits)

    total_bits = len(input_blocks) * 5038
    print(f"Decoded BER (excluding termination): {total_errors}/{total_bits} = {total_errors / total_bits:.6f}")
    print(f"Block errors: {block_errors}/{len(input_blocks)}")
    print("Coded-bit errors and decoded-bit errors use different lengths; do not compare raw totals.")
    return input_blocks, decoded_blocks


def main():
    """ตรวจพื้นฐานก่อนแสดงผลการทดลองทั้งสี่กรณี."""
    check_decoder()
    run_channel_example(use_turbulence=False, use_poisson=False)
    run_channel_example(use_turbulence=False, use_poisson=True)
    run_channel_example(use_turbulence=True, use_poisson=False)
    run_channel_example(use_turbulence=True, use_poisson=True)


if __name__ == "__main__":
    main()
