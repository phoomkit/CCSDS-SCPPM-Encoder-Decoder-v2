"""Step 8: ถอดส่วน frame ตาม EncoderDecoder_FirstPart (1).py.

รับ blocks หลัง Viterbi แล้วคืนข้อมูล frame โดยคงลำดับบิตเดิม
คงการตรวจภายในและพฤติกรรม ASM_Decoder ของต้นฉบับ ไม่มีการ print รายงาน
"""

import numpy as np


block_size = 5006
ASM_Value = 0x1ACFFC1D


def remove_terminationBit(blocks):
    """นำ termination สองบิตท้ายออกจากแต่ละ block หลัง Viterbi."""
    return [np.asarray(block, dtype=np.uint8)[:-2] for block in blocks]


def check_crc(blocks):
    """แยก CRC ท้าย 32 บิต และคำนวณเทียบด้วยกติกาเดียวกับต้นฉบับ."""
    data_blocks = []
    crc_ok = []
    crc_poly = (1 << 29) | (1 << 18) | (1 << 14) | (1 << 3) | 1

    for block in blocks:
        block = np.asarray(block, dtype=np.uint8)
        data_part = block[:-32]
        received_crc = block[-32:]
        reg = 0xFFFFFFFF

        for bit in data_part.tolist():
            feedback = ((reg >> 31) & 1) ^ bit
            reg = (reg << 1) & 0xFFFFFFFF
            if feedback:
                reg ^= crc_poly

        expected_crc = np.array(
            [(reg >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8
        )
        data_blocks.append(data_part)
        crc_ok.append(bool(np.array_equal(received_crc, expected_crc)))

    return data_blocks, crc_ok


def pseudo_randomize(blocks):
    """XOR ด้วยลำดับเดิมซ้ำอีกครั้งเพื่อย้อน randomizer: x ^ p ^ p = x."""
    if len(blocks) == 0:
        return []

    p = [1] * 8
    while len(p) < 255:
        p.append(p[-1] ^ p[-3] ^ p[-5] ^ p[-8])

    period = np.array(p, dtype=np.uint8)
    prn = np.resize(period, block_size)
    return [np.asarray(block, dtype=np.uint8) ^ prn[:len(block)] for block in blocks]


def Deslicer(blocks):
    """ต่อ blocks ตามลำดับเดิม โดยยังมี padding อยู่ท้าย stream."""
    return np.concatenate(blocks)


def ASM_Decoder(stream, frame_bits, crc_ok):
    """แยก frame ที่ตำแหน่งคาดไว้ ตามพฤติกรรมไฟล์ต้นฉบับ.

    frame_bits คือความยาวข้อมูลจริงที่ตกลงกัน ไม่รวม ASM และ padding
    ยังไม่ใช่การค้นหา synchronization เมื่อไม่รู้ตำแหน่งเริ่ม frame
    คืน tuples (ข้อมูล frame, พบ ASM หรือไม่, CRC ของ blocks ที่เกี่ยวข้องผ่านหรือไม่)
    """
    asm = np.unpackbits(np.array([ASM_Value], dtype=">u4").view(np.uint8))
    frame_len = 32 + frame_bits
    frames = []
    pos = 0

    while pos + frame_len <= len(stream):
        asm_found = bool(np.array_equal(stream[pos:pos + 32], asm))
        first_block = pos // block_size
        last_block = (pos + frame_len - 1) // block_size
        valid = all(crc_ok[first_block:last_block + 1])
        if not asm_found and valid:
            break
        frames.append((stream[pos + 32:pos + frame_len], asm_found, valid))
        pos += frame_len

    return frames


def decode_first_part(blocks, frame_bits):
    """ถอดหลัง Viterbi ครบทุกขั้น โดยเก็บผลตรวจภายในเหมือนไฟล์ต้นฉบับ."""
    blocks = remove_terminationBit(blocks)
    blocks, crc_ok = check_crc(blocks)
    blocks = pseudo_randomize(blocks)
    stream = Deslicer(blocks)
    return ASM_Decoder(stream, frame_bits, crc_ok), crc_ok
