"""SCPPM encoder แบบอัตรา 1/3 และ 4-PPM ตาม CCSDS 142.0-B-1 ข้อ 3.8."""

import numpy as np


INPUT_BITS = 5_040
CODE_BITS = 15_120
PPM_SYMBOLS = 7_560


def _validated_bits(input_bits, expected_length, name):
    """ตรวจสัญญาอินพุตบิตก่อนแปลงเป็น uint8 เพื่อไม่ให้ค่าผิดถูกตัดหรือ wrap."""
    try:
        bits = np.asarray(input_bits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} ต้องเป็นอาร์เรย์หนึ่งมิติของบิต 0/1") from exc

    if bits.ndim != 1:
        raise ValueError(f"{name} ต้องมี 1 มิติ แต่ได้รับ {bits.ndim} มิติ")
    if bits.size != expected_length:
        raise ValueError(
            f"{name} ต้องยาว {expected_length} บิต แต่ได้รับ {bits.size} บิต"
        )
    is_integer = np.issubdtype(bits.dtype, np.integer)
    is_boolean = np.issubdtype(bits.dtype, np.bool_)
    if not is_integer and not is_boolean:
        raise ValueError(f"{name} ต้องมี dtype เป็น integer หรือ bool")

    for bit in bits:
        if bit != 0 and bit != 1:
            raise ValueError(f"{name} ต้องมีเฉพาะค่า 0 หรือ 1")

    # ตรวจค่าครบแล้วจึงแปลงชนิดข้อมูล โดยสร้าง array ใหม่
    return np.array(bits, dtype=np.uint8)


def convolutional_encode(input_bits):
    """เข้ารหัส 5,040 บิตเป็น 15,120 บิตด้วย generator [5, 7, 7].

    หน่วยความจำเริ่มที่ 00 ทุก block และคำนวณ output ก่อนเลื่อน memory
    โดยเก็บ output 1, 2, 3 ของแต่ละ input bit เรียงติดกัน ไม่มี puncturing
    """
    input_bits = _validated_bits(input_bits, INPUT_BITS, "input_bits")

    # previous_1/previous_2 คือบิตย้อนหลัง 1/2 ตำแหน่งตามลำดับ
    previous_1 = 0
    previous_2 = 0
    output = []

    for current in input_bits:
        # ^ คือ XOR: บิตต่างกันได้ 1 บิตเหมือนกันได้ 0
        bit_1 = current ^ previous_2
        bit_2 = current ^ previous_1 ^ previous_2
        bit_3 = current ^ previous_1 ^ previous_2

        # เพิ่มทีละบิตตามลำดับ generator 1, 2, 3 ของ input บิตนี้
        output.append(bit_1)
        output.append(bit_2)
        output.append(bit_3)

        # คำนวณ output ให้เสร็จก่อนเลื่อน memory
        # ต้องเก็บ previous_1 เดิมลง previous_2 ก่อนแทนด้วย current
        previous_2 = previous_1
        previous_1 = current

    return np.array(output, dtype=np.uint8)


def code_interleave(coded_bits):
    """สลับตำแหน่ง 15,120 บิตด้วย l[j] = h[pi(j)].

    ทิศทางนี้สำคัญ: ตำแหน่ง output j อ่านจาก input ตำแหน่ง
    pi(j) = (11*j + 210*j^2) mod 15120 และไม่ได้เขียนทับ input
    """
    coded_bits = _validated_bits(coded_bits, CODE_BITS, "coded_bits")
    output = []

    for j in range(CODE_BITS):
        # j เป็น Python int จึงคำนวณพจน์กำลังสองได้โดยไม่ล้นแบบ int32
        position = (11 * j + 210 * j * j) % CODE_BITS

        # output ตำแหน่ง j อ่าน input ตำแหน่ง position (ไม่ใช่ทิศกลับกัน)
        bit = coded_bits[position]
        output.append(bit)

    return np.array(output, dtype=np.uint8)


def accumulate(interleaved_bits):
    """สะสม XOR จำนวน 15,120 บิต โดย reset state เป็น 0 ทุก block.

    ผลคือ n[0] = l[0] และ n[j] = n[j-1] XOR l[j] เมื่อ j >= 1
    """
    interleaved_bits = _validated_bits(
        interleaved_bits, CODE_BITS, "interleaved_bits"
    )
    state = 0
    output = []

    for bit in interleaved_bits:
        # state เดิมคือ output ก่อนหน้า เก็บผลใหม่หลัง XOR กับ input บิตนี้
        state = state ^ bit
        output.append(state)

    return np.array(output, dtype=np.uint8)


def ppm_symbol_map(accumulated_bits):
    """จับบิตทีละ 2 บิตเป็น 7,560 สัญลักษณ์ 4-PPM ค่า 0..3.

    รักษาลำดับเดิมโดยบิตแรกเป็น MSB: symbol = (first << 1) | second
    """
    accumulated_bits = _validated_bits(
        accumulated_bits, CODE_BITS, "accumulated_bits"
    )
    symbols = []

    # เริ่มอ่านที่ตำแหน่ง 0, 2, 4, ... ครั้งละหนึ่งคู่บิต
    for j in range(0, CODE_BITS, 2):
        first_bit = int(accumulated_bits[j])
        second_bit = int(accumulated_bits[j + 1])

        # << 1 เลื่อนบิตแรกไปเป็น MSB แล้วใช้ OR รวมกับบิตที่สอง
        # 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3
        symbol = (first_bit << 1) | second_bit
        symbols.append(symbol)

    return np.array(symbols, dtype=np.uint8)


def encode_scppm(input_bits):
    """เรียกสี่บล็อกตามลำดับและคืน 7,560 สัญลักษณ์ 4-PPM."""
    coded_bits = convolutional_encode(input_bits)
    interleaved_bits = code_interleave(coded_bits)
    accumulated_bits = accumulate(interleaved_bits)
    symbols = ppm_symbol_map(accumulated_bits)
    return symbols
