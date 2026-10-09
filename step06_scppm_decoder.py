"""Step 6: SCPPM hard-decision receiver สำหรับ rate 1/3 และ 4-PPM.

รับหนึ่ง block ที่รู้ขอบเขต slot/symbol แล้ว: 30,240 slots -> 5,040 bits
ใช้ PPM hard decision, inverse accumulator, deinterleaver และ Viterbi
ผลยังรวม CRC/termination; decoder ไม่ได้รับบิตต้นฉบับหรือค่า noise จริง
"""

import numpy as np


INPUT_BITS = 5040
CODE_BITS = 15120
RECEIVED_SLOTS = 30240


def _check_bits(bits):
    """ตรวจบิตระหว่างขั้น ก่อนแปลงเป็น uint8 เพื่อป้องกันค่าผิดถูก wrap."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size != CODE_BITS:
        raise ValueError("ต้องเป็น array หนึ่งมิติยาว 15120 บิต")
    if not (np.issubdtype(bits.dtype, np.integer) or bits.dtype == np.bool_):
        raise ValueError("บิตต้องเป็น integer หรือ bool")
    for bit in bits:
        if bit != 0 and bit != 1:
            raise ValueError("ค่าบิตต้องเป็น 0 หรือ 1")
    return np.array(bits, dtype=np.uint8)


def ppm_hard_demapper(received_slots):
    """เลือก slot สูงสุดในแต่ละกลุ่ม 4 slots แล้วคืนบิตแบบ MSB first.

    รับจำนวนโฟตอน หรือค่าความเข้มไม่ติดลบสำหรับการทดลองที่ปิด Poisson
    ถ้าค่าสูงสุดเสมอกันจะเลือกตำแหน่งแรก ไม่ได้หมายความว่ารู้บิตแน่นอน
    """
    received_slots = np.asarray(received_slots)
    if received_slots.ndim != 1 or received_slots.size != RECEIVED_SLOTS:
        raise ValueError("received_slots ต้องเป็น array หนึ่งมิติยาว 30240 slots")
    is_integer = np.issubdtype(received_slots.dtype, np.integer)
    is_float = np.issubdtype(received_slots.dtype, np.floating)
    is_boolean = received_slots.dtype == np.bool_
    if not (is_integer or is_float or is_boolean):
        raise ValueError("received_slots ต้องเป็นตัวเลขจริง")
    if not np.all(np.isfinite(received_slots)) or np.any(received_slots < 0):
        raise ValueError("received_slots ต้องไม่ติดลบ และไม่มี NaN/Infinity")

    output = []
    for start in range(0, RECEIVED_SLOTS, 4):
        four_slots = received_slots[start:start + 4]
        symbol = int(np.argmax(four_slots))

        # ย้อน mapping เดิม: 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
        first_bit = (symbol >> 1) & 1
        second_bit = symbol & 1
        output.append(first_bit)
        output.append(second_bit)

    return np.array(output, dtype=np.uint8)


def inverse_accumulator(accumulated_bits):
    """ย้อน accumulator โดย XOR บิตรับปัจจุบันกับบิตรับก่อนหน้า.

    เริ่ม previous_bit = 0 ทุก block; ไม่บังคับบิตสุดท้ายเป็น 0
    บิตรับที่ผิดหนึ่งตำแหน่งอาจทำให้ผลขั้นนี้ผิดสองตำแหน่งติดกัน
    """
    accumulated_bits = _check_bits(accumulated_bits)
    previous_bit = 0
    output = []

    for current_bit in accumulated_bits:
        bit = current_bit ^ previous_bit
        output.append(bit)
        # เก็บ input ก่อนหน้า ไม่ใช่ output ก่อนหน้า
        previous_bit = current_bit

    return np.array(output, dtype=np.uint8)


def code_deinterleave(interleaved_bits):
    """คืนบิตไปตำแหน่งเดิม: coded_bits[pi(j)] = interleaved_bits[j]."""
    interleaved_bits = _check_bits(interleaved_bits)
    coded_bits = np.zeros(CODE_BITS, dtype=np.uint8)

    for j in range(CODE_BITS):
        position = (11 * j + 210 * j * j) % CODE_BITS
        coded_bits[position] = interleaved_bits[j]

    return coded_bits


def viterbi_decode(coded_bits):
    """ถอดรหัส [5,7,7] ด้วย Viterbi แบบ hard decision และ Hamming distance.

    state = (previous_1 << 1) | previous_2 จึงมี 4 states: 00, 01, 10, 11
    เลือกเส้นทางที่มีจำนวนบิตไม่ตรงกับข้อมูลรับน้อยที่สุด
    เริ่ม state 00 และจบ state 00 ตาม termination 00 ของต้นทาง
    """
    coded_bits = _check_bits(coded_bits)

    # สร้าง trellis: แต่ละ state รับ input ได้ 0 หรือ 1
    next_states = np.zeros((4, 2), dtype=np.uint8)
    expected_bits = np.zeros((4, 2, 3), dtype=np.uint8)

    for state in range(4):
        previous_1 = (state >> 1) & 1
        previous_2 = state & 1

        for input_bit in range(2):
            expected_bits[state, input_bit, 0] = input_bit ^ previous_2
            expected_bits[state, input_bit, 1] = input_bit ^ previous_1 ^ previous_2
            expected_bits[state, input_bit, 2] = input_bit ^ previous_1 ^ previous_2
            next_states[state, input_bit] = (input_bit << 1) | previous_1

    # ตอนเริ่มต้นไปได้เฉพาะ state 00 ส่วน state อื่นยังไม่มีเส้นทาง
    path_costs = [0, float("inf"), float("inf"), float("inf")]
    saved_previous_states = np.zeros((INPUT_BITS, 4), dtype=np.uint8)

    # ข้อมูลรับทุก 3 บิตมาจาก input เดิม 1 บิต
    for time_index in range(INPUT_BITS):
        start = time_index * 3
        received = coded_bits[start:start + 3]
        new_costs = [float("inf")] * 4

        for state in range(4):
            if path_costs[state] == float("inf"):
                continue

            for input_bit in range(2):
                next_state = int(next_states[state, input_bit])

                # Hamming distance: นับว่าผลที่คาดไว้ต่างจากข้อมูลรับกี่บิต
                branch_cost = 0
                for bit_index in range(3):
                    if expected_bits[state, input_bit, bit_index] != received[bit_index]:
                        branch_cost += 1

                total_cost = path_costs[state] + branch_cost
                if total_cost < new_costs[next_state]:
                    new_costs[next_state] = total_cost
                    saved_previous_states[time_index, next_state] = state
                # ถ้า cost เท่ากัน เก็บเส้นทางแรกไว้เพื่อให้ผลทำซ้ำได้

        path_costs = new_costs

    # Traceback: เดินย้อนเส้นทางจาก state สุดท้าย 00
    decoded_bits = np.zeros(INPUT_BITS, dtype=np.uint8)
    state = 0
    for time_index in range(INPUT_BITS - 1, -1, -1):
        # บิตสูงของ state หลังรับ input คือ input_bit ของเวลานั้น
        decoded_bits[time_index] = (state >> 1) & 1
        state = int(saved_previous_states[time_index, state])

    return decoded_bits


def decode_scppm(received_slots):
    """รับ r(t) ครบหนึ่ง block แล้วคืน 5040 บิตรวม CRC และ termination."""
    accumulated_bits = ppm_hard_demapper(received_slots)
    interleaved_bits = inverse_accumulator(accumulated_bits)
    coded_bits = code_deinterleave(interleaved_bits)
    decoded_bits = viterbi_decode(coded_bits)
    return decoded_bits
