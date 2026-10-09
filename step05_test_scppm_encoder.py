"""สคริปต์ตรวจรับและวาดกราฟ prefix ของ SCPPM encoder."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from step02_scppm_encoder import (
    CODE_BITS,
    INPUT_BITS,
    PPM_SYMBOLS,
    accumulate,
    code_interleave,
    convolutional_encode,
    encode_scppm,
    ppm_symbol_map,
)


# ให้ข้อความภาษาไทยแสดงได้บน Windows console ที่ค่าเริ่มต้นอาจเป็น cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _expect_value_error(callable_, description: str) -> None:
    """ยืนยันว่าอินพุตผิดสัญญาถูกปฏิเสธด้วย ValueError."""
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(f"ไม่พบ ValueError: {description}")


def run_acceptance_checks() -> tuple[np.ndarray, ...]:
    """รันเกณฑ์ตรวจรับจากพิมพ์เขียวและคืนข้อมูลตัวอย่างสำหรับ plot."""
    # 1) อินพุตศูนย์ทั้ง block ต้องให้ศูนย์ตลอดทุก stage
    zeros = np.zeros(INPUT_BITS, dtype=np.uint8)
    h0 = convolutional_encode(zeros)
    l0 = code_interleave(h0)
    n0 = accumulate(l0)
    q0 = ppm_symbol_map(n0)
    assert h0.shape == l0.shape == n0.shape == (CODE_BITS,)
    assert q0.shape == (PPM_SYMBOLS,)
    assert h0.dtype == l0.dtype == n0.dtype == q0.dtype == np.uint8
    assert not np.any(h0) and not np.any(l0) and not np.any(n0)
    assert not np.any(q0)

    # 2) impulse ที่บิตแรกตรวจลำดับ output ของ generator [5, 7, 7]
    impulse = zeros.copy()
    impulse[0] = 1
    hi = convolutional_encode(impulse)
    np.testing.assert_array_equal(
        hi[:9], np.fromiter((int(x) for x in "111011111"), dtype=np.uint8)
    )
    assert not np.any(hi[9:])

    # 3) interleaver ต้องเป็น permutation และอ่าน h[pi(j)] ตามทิศทางมาตรฐาน
    j = np.arange(CODE_BITS, dtype=np.int64)
    pi = (11 * j + 210 * j * j) % CODE_BITS
    np.testing.assert_array_equal(pi[:6], [0, 221, 862, 1923, 3404, 5305])
    assert np.unique(pi).size == CODE_BITS
    one_hot = np.zeros(CODE_BITS, dtype=np.uint8)
    one_hot[221] = 1
    one_hot_l = code_interleave(one_hot)
    assert one_hot_l[1] == 1 and np.count_nonzero(one_hot_l) == 1

    # 4) accumulator ต้องใช้ output ก่อนหน้าและ reset state ทุกครั้ง
    accumulator_input = np.zeros(CODE_BITS, dtype=np.uint8)
    accumulator_input[:7] = [1, 0, 1, 1, 0, 0, 1]
    expected_prefix = np.array([1, 1, 0, 1, 1, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(accumulate(accumulator_input)[:7], expected_prefix)
    np.testing.assert_array_equal(accumulate(accumulator_input)[:7], expected_prefix)

    # 5) mapper ใช้บิตแรกเป็น MSB: 00, 01, 10, 11 -> 0, 1, 2, 3
    mapper_input = np.zeros(CODE_BITS, dtype=np.uint8)
    mapper_input[:8] = [0, 0, 0, 1, 1, 0, 1, 1]
    np.testing.assert_array_equal(ppm_symbol_map(mapper_input)[:4], [0, 1, 2, 3])

    # 6) full chain ต้อง deterministic, ไม่แก้ input และ output อยู่ในช่วง 0..3
    # ชุดนี้เป็น synthetic encoder input; ไม่ได้อ้างว่า CRC 32 บิตถูกคำนวณจริง
    rng = np.random.default_rng(1420)
    f = rng.integers(0, 2, size=INPUT_BITS, dtype=np.uint8)
    f[-2:] = 0  # termination bits ที่ upstream แนบมาแล้ว
    original = f.copy()
    h = convolutional_encode(f)
    l = code_interleave(h)
    n = accumulate(l)
    q = ppm_symbol_map(n)
    np.testing.assert_array_equal(q, encode_scppm(f))
    np.testing.assert_array_equal(q, encode_scppm(f))
    np.testing.assert_array_equal(f, original)
    assert q.shape == (PPM_SYMBOLS,) and q.dtype == np.uint8
    assert np.all(q <= 3)

    # 7) ความยาว ค่า และ dtype ที่ผิดต้องถูกปฏิเสธ
    _expect_value_error(lambda: convolutional_encode([0, 1]), "ความยาวผิด")
    invalid_value = zeros.copy()
    invalid_value[100] = 2
    _expect_value_error(lambda: convolutional_encode(invalid_value), "มีค่าที่ไม่ใช่บิต")
    _expect_value_error(
        lambda: convolutional_encode(zeros.astype(np.float64)), "dtype เป็น float"
    )

    print("PASS: ผ่านเกณฑ์ตรวจรับ SCPPM encoder ทั้งหมด")
    print(f"f: {f.shape} {f.dtype}")
    print(f"h: {h.shape} {h.dtype}")
    print(f"l: {l.shape} {l.dtype}")
    print(f"n: {n.shape} {n.dtype}")
    print(f"q: {q.shape} {q.dtype}, range={int(q.min())}..{int(q.max())}")
    return f, h, l, n, q


def plot_prefix(stages: tuple[np.ndarray, ...], prefix: int = 64) -> None:
    """วาดเฉพาะ prefix ของแต่ละ stage เพื่อให้เห็นการแปลงโดยไม่ตีความเป็น BER."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ไม่พบ Matplotlib: ติดตั้งด้วย 'python -m pip install matplotlib'"
        ) from exc

    f, h, l, n, q = stages
    bit_stages = (("f: encoder input", f), ("h: convolutional", h),
                  ("l: interleaved", l), ("n: accumulated", n))
    figure, axes = plt.subplots(5, 1, figsize=(12, 10), constrained_layout=True)

    for axis, (label, data) in zip(axes[:4], bit_stages):
        count = min(prefix, data.size)
        axis.step(np.arange(count), data[:count], where="post")
        axis.set(title=label, ylabel="bit", yticks=[0, 1], ylim=(-0.15, 1.15))
        axis.grid(alpha=0.25)

    symbol_count = min(prefix // 2, q.size)
    axes[4].step(np.arange(symbol_count), q[:symbol_count], where="mid")
    axes[4].set(
        title="q: 4-PPM symbols (integer)", xlabel="index", ylabel="symbol",
        yticks=[0, 1, 2, 3], ylim=(-0.25, 3.25),
    )
    axes[4].grid(alpha=0.25)
    plt.show()


def main() -> None:
    """รันการตรวจรับ และเปิดกราฟเมื่อระบุ --plot."""
    parser = argparse.ArgumentParser(description="ตรวจรับ SCPPM encoder")
    parser.add_argument("--plot", action="store_true", help="แสดงกราฟ prefix ของทุก stage")
    parser.add_argument("--prefix", type=int, default=64, help="จำนวนบิต prefix ที่แสดง")
    args = parser.parse_args()

    stages = run_acceptance_checks()
    if args.plot:
        if args.prefix <= 0:
            raise ValueError("--prefix ต้องมากกว่า 0")
        plot_prefix(stages, args.prefix)


if __name__ == "__main__":
    main()
