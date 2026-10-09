import numpy as np

Nbits= 10012
np.random.seed(3)
data_bit = np.random.randint(0,2,Nbits)

#PARAMETER
block_size = 5006  #code rate 1/3 ใช้ 5006, 1/2 ใช่ 7526, 2/3 ใช่ 10046
ASM_Value = 0x1ACFFC1D

def ASM_Encoder(data_bits):

    
    ASM_bit_array = np.unpackbits(np.array([ASM_Value], dtype='>u4').view(np.uint8))    #แปลงให้เป็น numpy array
    
    return np.concatenate((ASM_bit_array,data_bits))

def Slicer(data_bit):

 
    remainder = len(data_bit) % block_size
    
    if remainder != 0:
        zero_array = block_size - remainder
        # 2. Append zeros to the end of the original array
        data_bit = np.concatenate((data_bit, np.zeros(zero_array, dtype=data_bit.dtype)))
    num_block = len(data_bit) // block_size
    return np.split(data_bit, num_block)

def pseudo_randomize(blocks):

    if len(blocks) == 0:
        return []
    # one period of the sequence: p[i+8] = p[i+7] ^ p[i+5] ^ p[i+3] ^ p[i]
    p = [1] * 8
    while len(p) < 255:
        p.append(p[-1] ^ p[-3] ^ p[-5] ^ p[-8])
    period = np.array(p, dtype=np.uint8)
 

    prn = np.resize(period, block_size)               # repeat every 255 bits as far as needed
    return [np.asarray(block, dtype=np.uint8) ^ prn[:len(block)] for block in blocks]
    
def attach_crc(blocks):
    """Append 32 CRC bits to the end of each block."""

    out = []
    CRC_POLY = (1 << 29) | (1 << 18) | (1 << 14) | (1 << 3) | 1      # = 0x20044009
    for block in blocks:
        reg = 0xFFFFFFFF  # Register preset to all ones

        for bit in np.asarray(block, dtype=np.uint8).tolist():
            feedback = ((reg >> 31) & 1) ^ bit
            reg = (reg << 1) & 0xFFFFFFFF

            if feedback:
                reg ^= CRC_POLY

        crc = np.array([(reg >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
        out.append(np.concatenate([np.asarray(block, dtype=np.uint8), crc]))   # inside the loop
        
    return out      # return the list

def attach_terminationBit(blocks):
    terminate_bit = np.array([0, 0], dtype=np.uint8)
    # This creates a new list with the updated arrays and returns it
    return [np.concatenate((block, terminate_bit)) for block in blocks]    

data_asm = ASM_Encoder(data_bit)
data_sliced = Slicer(data_asm)
data_prn = pseudo_randomize(data_sliced)
data_crc = attach_crc(data_prn)
data = attach_terminationBit(data_crc)
print(data)