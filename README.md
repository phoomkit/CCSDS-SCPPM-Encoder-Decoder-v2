# SCPPM Encoder–Decoder v.2 (Python + NumPy)

เวอร์ชันนี้เน้นความเข้าใจ: ทั้งสี่บล็อกใช้ `for`, ตัวแปรแยกทีละขั้น และคอมเมนต์ภาษาไทย
ใช้ `^` สำหรับ XOR, `<<` สำหรับเลื่อนบิต และ `|` สำหรับ OR
ช้ากว่าเวอร์ชัน NumPy vectorization แต่คงลำดับบิตและผลลัพธ์เดิม

## ชื่อไฟล์

- `step01_input_preparation.py` - ไฟล์ต้นฉบับส่วนแรก เนื้อหาไม่ได้แก้ไข
- `step02_scppm_encoder.py` - ฟังก์ชัน SCPPM encoder ทั้งสี่ขั้น
- `step03_generate_ppm_output.py` - สร้างผล PPM จากข้อมูลจริงลง `ppm_output.txt`
- `step04_plot_ppm_output.py` - เพิ่ม atmospheric/Poisson noise และแสดงกราฟทั้งสาม blocks
- `step05_test_scppm_encoder.py` - ตรวจความถูกต้องของ encoder
- `step06_scppm_decoder.py` - PPM hard decision, inverse accumulator, deinterleaver และ Viterbi
- `step07_test_scppm_decoder.py` - ทดสอบ decoder กับ Noise แต่ละกรณี
- `step08_frame_decoder.py` - ถอด termination, CRC, randomizer, slicer และ ASM
- `step09_run_full_system.py` - รันระบบครบ แสดงกราฟสัญญาณและ BER curve
- `DECODER_README.md` - หลักการและวิธีใช้ decoder
- `FULL_SYSTEM_README.md` - วิธีรันระบบเต็มและอ่านกราฟ
- `README.md` - คู่มือฉบับนี้

## เริ่มอ่านและรัน

1. อ่าน `step01_input_preparation.py` เพื่อดูการเตรียมข้อมูลจนได้ termination bits
2. อ่านฟังก์ชันแต่ละตัวใน `step02_scppm_encoder.py` ตามลำดับ convolutional, interleaver, accumulator และ mapper
3. `_validated_bits` เป็นฟังก์ชันตรวจอินพุตร่วมกัน แยกไว้จากขั้นตอนเข้ารหัส
4. `step03_generate_ppm_output.py` และ `step04_plot_ppm_output.py` ใช้สร้างผลลัพธ์และกราฟ
5. `step05_test_scppm_encoder.py` เป็นสคริปต์ทดสอบ encoder สามารถอ่านทีหลังได้

วางไฟล์ทั้งหมดในโฟลเดอร์เดียวกัน แล้วรันจากโฟลเดอร์นั้น:

```powershell
python -m pip install numpy
python step05_test_scppm_encoder.py
python step07_test_scppm_decoder.py
python step09_run_full_system.py
```

`step01_input_preparation.py` มีเนื้อหาตรงกับไฟล์ต้นฉบับ `EncoderDecoder_FirstPart.py`

## ขอบเขต encoder

โค้ดนี้ทำเฉพาะ encoder 4 บล็อกตาม CCSDS 142.0-B-1 ข้อ 3.8:

1. Convolutional encoder อัตรา `1/3`, generator `[5, 7, 7]`
2. Code interleaver ขนาด 15,120 บิต
3. Accumulator แบบ XOR
4. Mapper สำหรับ `M = 4` โดยบิตแรกเป็น MSB

## สัญญาอินพุต/เอาต์พุต

- อินพุต `encode_scppm`: list หรือ NumPy array หนึ่งมิติ ยาว 5,040 บิต
- ทุกสมาชิกเป็น integer/bool ค่า `0` หรือ `1`
- อินพุตรวม randomized information 5,006 บิต, CRC 32 บิต และ termination `00` แล้ว
- โมดูลไม่สร้างหรือตรวจ CRC และไม่เติม/ตัด termination bits
- เอาต์พุต: `np.ndarray(dtype=np.uint8)` ยาว 7,560 ค่าอยู่ในช่วง 0 ถึง 3
- ทุกฟังก์ชันสร้าง output ใหม่และเริ่ม state ใหม่ทุก block
- interleaver คำนวณตำแหน่งด้วย Python `int` จาก `range` จึงไม่ล้นแบบ `int32`; ทิศทางยังเป็น `output[j] = input[pi(j)]`

## เรียกใช้งาน

```python
import numpy as np
from step02_scppm_encoder import encode_scppm

f = np.zeros(5040, dtype=np.uint8)  # ตัวอย่างเท่านั้น; upstream ต้องเตรียม CRC/termination
q = encode_scppm(f)
print(q.shape, q.dtype)  # (7560,) uint8
```

สามารถเรียกแต่ละฟังก์ชันแยกเพื่อดูผลระหว่างทางได้:

```python
from step02_scppm_encoder import (
    convolutional_encode, code_interleave, accumulate, ppm_symbol_map
)

h = convolutional_encode(f)
l = code_interleave(h)
n = accumulate(l)
q = ppm_symbol_map(n)
```

ต้องมี NumPy ส่วน Matplotlib ใช้เฉพาะเมื่อขอแสดงกราฟ:

```powershell
python -m pip install numpy matplotlib
python step05_test_scppm_encoder.py
python step05_test_scppm_encoder.py --plot --prefix 64
```

สคริปต์แรกจะรันเกณฑ์ตรวจรับและแสดง `PASS` พร้อม shape/dtype ของทุก stage ส่วนคำสั่งที่สองจะแสดงกราฟ prefix โดยไม่เปรียบเทียบ BER เพราะยังไม่มี decoder/recovered message

## กราฟ 4-PPM

ติดตั้ง Matplotlib แล้วเรียก `step04_plot_ppm_output.py` เพื่อดูทั้งหมายเลข symbol และตำแหน่ง pulse ใน 4 slots:

```powershell
python -m pip install matplotlib
python step04_plot_ppm_output.py
```

โปรแกรมจะแสดง 16 symbols แรกของ Block 1, Block 2 และ Block 3 ในหน้าต่างเดียว
แต่ละแถวมีกราฟค่า symbol ทางซ้าย และกราฟ `x(t)`, `n(t)`, `r(t)` ทางขวา
โปรแกรมใช้ `plt.show()` เพื่อแสดงกราฟใน VS Code และไม่มีคำสั่งบันทึกรูป

ใน `plot_all_blocks()` สามารถใส่ `#` หน้าบรรทัดเรียก
`atmospheric_turbulence_noise(x_t)` หรือ `poisson_noise(x_after_atmosphere)`
เพื่อปิด noise แต่ละชนิดได้

เมื่อนำโค้ดไปใช้ใน Jupyter Notebook โปรแกรมจะหาโฟลเดอร์ `CCSDS102` ให้อัตโนมัติ
และแสดงกราฟใน output ของ Notebook โดยไม่ต้องใช้ตัวแปร `__file__`

```python
%matplotlib inline
%run step04_plot_ppm_output.py
```
