# SCPPM Decoder — hard decision + Viterbi

เวอร์ชันนี้เน้นให้ไล่อ่านได้ทีละฟังก์ชัน เป็น receiver แบบง่ายสำหรับ encoder เดิม
ใช้ rate 1/3, generator [5,7,7], memory 2 บิต และ 4-PPM
ไม่ได้ใช้ iterative BCJR และไม่ได้อ้างประสิทธิภาพเทียบเท่า iterative SCPPM receiver

## ไฟล์ที่เพิ่ม

- `step06_scppm_decoder.py`: ฟังก์ชันถอดรหัส 4 ขั้นและ wrapper
- `step07_test_scppm_decoder.py`: ตรวจความถูกต้องและทดลอง noise 4 กรณี
- `DECODER_README.md`: วิธีใช้งานและหลักการสำหรับเตรียมนำเสนอ

โค้ดใช้ NumPy และใช้ฟังก์ชัน noise ที่มีอยู่ใน Step 04
ไม่ต้องติดตั้ง Matplotlib หากรันเฉพาะ Step 07 เพราะไม่ได้เรียกกราฟ
ไม่มีการเปลี่ยน Step 01–05 หรือการตั้งค่า noise ในไฟล์เหล่านั้น

## จุดเข้าและจุดออก

```text
r_t: 30,240 slots
    -> ppm_hard_demapper       -> 15,120 accumulated bits
    -> inverse_accumulator     -> 15,120 interleaved bits
    -> code_deinterleave        -> 15,120 convolutional coded bits
    -> viterbi_decode           -> 5,040 decoded input bits
```

อินพุตเป็น array หนึ่งมิติ เรียง slots ตามเวลา มีค่าจำกัดและไม่ติดลบ
เมื่อเปิด Poisson ค่านี้คือ photon counts; เมื่อปิด Poisson ยอมรับความเข้มแบบ float ได้
รับครบ 30,240 slots ต่อ block ไม่ใช่ 320 slots ที่ตัดมา plot 80 symbols
หากมีข้อมูล shape `(7560, 4)` ให้เรียงกลับเป็นหนึ่งมิติด้วย `.reshape(-1)`
สมมติว่ารู้ขอบเขต block/symbol/slot แล้ว งานนี้ยังไม่ทำ synchronization

Output เป็น `np.ndarray` dtype `uint8`, shape `(5040,)` ประกอบด้วย:

```text
ข้อมูลก่อนเข้า encoder 5,006 บิต | CRC 32 บิต | termination 00
```

ยังไม่เอา termination ออก ไม่ตรวจ CRC ไม่ de-randomize และไม่ดึง ASM/payload กลับ
ทั้ง 5,040 บิตควรตรงกับ `data[block_index]` จาก Step 01 เมื่อถอดสำเร็จ

## วิธีเรียกจาก r_t ที่คุณสร้างไว้

```python
from step06_scppm_decoder import decode_scppm

decoded_bits = decode_scppm(r_t)
print(decoded_bits.shape)  # (5040,)
print(decoded_bits[:32])
```

หรือเรียกทีละขั้นเพื่อศึกษา:

```python
from step06_scppm_decoder import (
    ppm_hard_demapper, inverse_accumulator, code_deinterleave, viterbi_decode
)

accumulated_bits = ppm_hard_demapper(r_t)
interleaved_bits = inverse_accumulator(accumulated_bits)
coded_bits = code_deinterleave(interleaved_bits)
decoded_bits = viterbi_decode(coded_bits)
```

ใน Step 07 จะเข้ารหัสและสร้าง noise ครบทั้ง block ก่อนส่งให้ decoder
การตัดบางส่วนมา plot ควรทำหลังได้ `r_t` เต็ม block แล้ว
ตัว decoder รับเฉพาะ `r_t` ไม่มีการใช้ noise จริงหรือข้อมูลต้นฉบับมาช่วยตัดสิน

## รันทดสอบ

เปิด terminal ในโฟลเดอร์ CCSDS102 แล้วรัน:

```powershell
python step07_test_scppm_decoder.py
```

หรือใช้ Jupyter:

```python
%run "C:/Users/phoom/OneDrive/Desktop/CCSDS102/step07_test_scppm_decoder.py"
```

เมื่อรันแล้วจะตรวจ inverse stages, round trip ไม่มี noise และการแก้ single-bit error
จากนั้นทดลอง 4 กรณี: ไม่มี noise, Poisson, turbulence, และสองอย่างร่วมกัน
ไม่มีการเปิดหน้าต่างกราฟหรือบันทึกภาพ

เลือกทดลองเองใน Cell ถัดไปหลัง `%run`:

```python
# เปิดทั้งสอง noise ที่ค่าเฉลี่ย 1 photon ต่อ pulse
input_blocks, decoded_blocks = run_channel_example(
    use_turbulence=True,
    use_poisson=True,
    signal_photons=1.0,
)

# ดูบิตต้นฉบับและบิตที่ถอดได้ของ block แรก
print(input_blocks[0][:32])
print(decoded_blocks[0][:32])
```

เปลี่ยน True/False เพื่อเปิด–ปิดแต่ละ noise แยกกัน
ภายใน Step 07 ยังมีบรรทัดเรียก noise แยกกันและค่าเริ่มต้นศูนย์เช่นเดิม
การเลือก noise ของ Step 07 แยกจากบรรทัดที่คุณ comment ไว้ในกราฟ Step 04
`signal_photons` เปลี่ยนค่าเฉลี่ยจำนวนโฟตอนต่อ pulse ก่อน fading ไม่เปลี่ยนบิตหรือ PPM order
ใช้ seed แยกตาม block และชนิด noise เพื่อทำซ้ำผลได้

## อธิบายแต่ละส่วน

1. **PPM hard demapper:** เลือก slot ที่สูงสุดในกลุ่ม 4 แล้วคืน 2 บิตแบบ MSB first
   เช่น `[0,1,5,0]` เลือก symbol 2 ได้บิต `10`
   เมื่อเสมอกันเลือก slot แรก รวมถึง `[0,0,0,0]` จะเลือก symbol 0
   นี่เป็นการตัดสินแบบง่ายที่มีอคติไปตำแหน่งแรก ไม่ใช่การรู้ข้อมูลที่หายไป
2. **Inverse accumulator:** บิตเดิมเท่ากับ input ปัจจุบัน XOR input ก่อนหน้า
   เริ่ม previous = 0; ไม่บังคับสถานะสุดท้ายของ accumulator เป็นศูนย์
3. **Deinterleaver:** คืนข้อมูลด้วย `coded[pi(j)] = interleaved[j]`
   ซึ่งย้อนทิศจาก encoder ที่ใช้ `interleaved[j] = coded[pi(j)]`
4. **Viterbi:** จำเส้นทางที่มีจำนวนบิตต่างจากข้อมูลรับน้อยที่สุดในแต่ละ state
   จากนั้นย้อนเส้นทาง (traceback) เพื่อกู้ input bits

Viterbi มี 4 states ตาม `(previous_1, previous_2)`:

| State | input 0: output -> next state | input 1: output -> next state |
|---|---|---|
| 00 | 000 -> 00 | 111 -> 10 |
| 01 | 111 -> 00 | 000 -> 10 |
| 10 | 011 -> 01 | 100 -> 11 |
| 11 | 100 -> 01 | 011 -> 11 |

Branch cost คือ Hamming distance เช่น คาด `111` แต่รับ `101` จะมี cost 1
Path cost คือผลรวม branch costs ตลอดเส้นทาง; หากเสมอเก็บเส้นทางแรก
เริ่มและจบที่ state 00 ตาม initialization และ termination ของ convolutional encoder
การบังคับ end state ทำให้สองบิตท้ายเป็น 00 เสมอ จึงไม่ใช้ tail สองบิตเป็นหลักฐานว่าถอดถูก
วิธีนี้หาเส้นทางระยะ Hamming ต่ำสุดหลัง hard decision ไม่ใช่การหา likelihood สูงสุดร่วมทั้ง Poisson channel และ SCPPM

## อ่านผลทดลอง

- `tied PPM symbols`: จำนวนกลุ่มที่ค่าสูงสุดเสมอกัน บอกจำนวนการตัดสินที่กำกวม
- `coded errors / 15120`: บิตผิดก่อน Viterbi เทียบกับ convolutional codeword จริง
- `decoded errors / 5038`: บิตผิดหลัง Viterbi เทียบข้อมูลเดิมรวม CRC แต่ไม่รวม termination
- `Decoded BER`: บิตผิดหลังถอด / จำนวนบิตที่ประเมินรวมทุก block
- `Block errors`: จำนวน blocks ที่มีบิตผิดอย่างน้อยหนึ่งบิต

ตัวหารก่อนและหลัง Viterbi ต่างกัน จึงไม่เปรียบเทียบจำนวน error ดิบโดยตรง
BER นี้วัดที่อินพุต/เอาต์พุตของ SCPPM ไม่ใช่ payload BER หลัง de-randomize

ผลตัวอย่างจากข้อมูล Step 01 จำนวน 3 blocks และ seeds ที่กำหนด:

| Noise | Mean signal photons/pulse | Decoded errors / 15114 bits | Block errors |
|---|---:|---:|---:|
| ปิดทั้งสอง | 1 | 0 | 0/3 |
| Poisson | 1 | 5886 | 3/3 |
| Turbulence | 1 | 0 | 0/3 |
| Poisson + turbulence | 1 | 6144 | 3/3 |
| Poisson + turbulence | 10 | 0 | 0/3 |

ผลนี้เป็นตัวอย่างรอบทดลอง ไม่ใช่ reference vectors หรือคำรับรอง BER ของระบบ
การไม่พบ error ใน 3 blocks ไม่ได้พิสูจน์ว่าโอกาสผิดเป็นศูนย์
ที่ 1 photon/pulse มีโอกาสไม่พบโฟตอนใน pulse slot มาก จึงเป็นงานยากสำหรับ hard decision
Turbulence อย่างเดียวในโมเดล gain บวกที่ใช้ จะไม่ย้ายตำแหน่ง pulse หรือเพิ่มแสงใน slot ศูนย์
จึงอาจไม่ทำให้ hard decision ผิด; ผลร่วมกับ Poisson ต่างออกไป

หากแก้ข้อมูลต้นทาง ฟังก์ชัน noise หรือ seed ตัวเลขการทดลองอาจเปลี่ยน
ทดลองหลาย seeds/blocks ก่อนสรุปเชิงสถิติ

## แนวทางพัฒนาต่อ

ตัวเลือกถัดไปคือ iterative soft decoding ร่วม PPM/accumulator กับ convolutional BCJR
ซึ่งใช้ likelihood และแลกเปลี่ยน extrinsic information แทนการตัดสินบิตทันที
อ้างอิงแนวทาง SCPPM: [JPL, Coded Modulation for the Deep-Space Optical Channel](https://ipnpr.jpl.nasa.gov/progress_report/42-161/161T.html)
ครั้งนี้ทำเฉพาะ simplified hard-decision receiver ตามแผนที่เลือก
