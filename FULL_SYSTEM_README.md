# แสดงข้อมูลก่อนและหลัง Decoder

รัน `step09_run_full_system.py` เพื่อแสดงกราฟระบบ 6 แถว และ BER curve อีกหนึ่งหน้าต่าง
ไฟล์นี้ใช้ Step 1, 2, 4, 6 และ 8 ที่อยู่ในโฟลเดอร์เดียวกัน

1. Original data: ข้อมูลต้นฉบับก่อนเข้ารหัส
2. Transmitted x(t): ข้อมูลที่ห่อและเข้ารหัสเป็น 4-PPM แล้ว ก่อนผ่าน Noise
3. After atmospheric turbulence: สัญญาณก่อนและหลังบรรยากาศซ้อนกัน
4. Before and after Poisson: สัญญาณก่อนและหลังการนับ photon ซ้อนกัน
5. Received signal: `r_t` ที่ป้อนเข้า SCPPM decoder จริง
6. Recovered data: ข้อมูลที่ถอดกลับครบทุกขั้น ซ้อนกับต้นฉบับ

BER curve ใช้แกน X เป็น mean signal photons/pulse และแกน Y เป็น payload BER
แบบ log scale โดยทดลองระดับ `[0.5, 1, 2, 3, 5, 7, 10]` ระดับละ 10 รอบ
หนึ่งจุดจึงใช้ payload รวม 100,120 บิต และสุ่ม Noise ใหม่ทุกครั้ง
ถ้าไม่พบ error ค่า BER ที่วัดได้คือ 0 ซึ่งวางบน log scale ไม่ได้
กราฟจะแสดงสามเหลี่ยมหัวลงที่ `1 / จำนวนบิต` เพื่อบอกว่าเป็นขีดจำกัดบน

ใช้ `plt.show()` ไม่มีการบันทึกรูปหรือข้อมูลอัตโนมัติ
ชื่อกราฟเป็นภาษาอังกฤษเพื่อแสดงได้โดยไม่ต้องติดตั้งฟอนต์ไทย

## รันใน VS Code / Jupyter

เปิดไฟล์ Step 9 แล้วกด Run Python File หรือใช้ notebook cell:

```python
%run "C:/Users/phoom/OneDrive/Desktop/CCSDS102/step09_run_full_system.py"
```

หลังรันสามารถดูข้อมูลจริงได้:

```python
original_data = result["original_data"]
recovered_data = result["recovered_data"]
print(recovered_data.shape)
print(recovered_data[:128])
```

ใช้ `%run` กับไฟล์แทนการคัดลอกทั้งไฟล์ลง cell เพราะตัวโหลด Step 1 ใช้ `__file__`
ถ้า environment ที่เลือกไม่มีไลบรารี ให้ติดตั้ง NumPy และ Matplotlib ใน environment นั้น

## ปรับการทดลอง

ค่าที่ต้นไฟล์ Step 9:

```python
USE_TURBULENCE = True
USE_POISSON = True
SIGNAL_PHOTONS = 10.0
BER_PHOTON_LEVELS = [0.5, 1, 2, 3, 5, 7, 10]
BER_TRIALS_PER_LEVEL = 10
NUMBER_OF_BITS = 128
NUMBER_OF_SYMBOLS = 64
PLOT_BLOCK = 0
```

ตั้ง True/False เพื่อเปิดปิด Noise แยกกัน ค่าเริ่มต้นเปิดทั้งคู่ที่ 10 photons/pulse
ใช้ Noise functions เดิมจาก Step 4 แต่สวิตช์ใน Step 9 เป็นของการทดลองนี้เอง
การ comment บรรทัดในฟังก์ชันวาดกราฟของ Step 4 ไม่เปลี่ยนสวิตช์ Step 9
ค่า seed ใช้ตาม Step 7 จึงทำซ้ำการทดลองได้

หลัง `%run` สามารถทดลองใน cell ถัดไปโดยไม่แก้ไฟล์:

```python
result = run_full_system(
    use_turbulence=True,
    use_poisson=True,
    signal_photons=1.0,
)
figure = plot_full_system(result)

ber_curve = calculate_ber_curve(
    [0.5, 1, 2, 3, 5, 7, 10],
    trials_per_level=10,
    use_turbulence=True,
    use_poisson=True,
)
ber_figure = plot_ber_curve(ber_curve)
```

เมื่อปิด Poisson กราฟ r(t) เป็นความเข้ม/ค่าเฉลี่ยโฟตอน ไม่ใช่จำนวนโฟตอนที่สุ่มตรวจพบ
`n_poisson` เป็นผลต่างจากสัญญาณก่อนเข้า Poisson ส่วน `r_t` ไม่ติดลบ

## การคืนข้อมูล

Step 8 ใช้ขั้นตอนจาก `EncoderDecoder_FirstPart (1).py`:
ตัด termination → ตรวจและแยก CRC → de-randomize → รวม blocks → ASM_Decoder
การตรวจภายในและผลคืนของ ASM_Decoder คงตามต้นฉบับ
ไม่มีรายงานแยก asm_ok, crc_ok, payload_errors หรือ BER ในหน้าจอและกราฟใหม่

ข้อมูลปัจจุบันคือหนึ่ง frame ยาว 10,012 บิต รู้ตำแหน่งเริ่ม frame และขอบเขต slots/blocks แล้ว
ความยาว frame เป็นค่าที่ตกลงไว้ทั้งสองฝั่ง ไม่ใช้การตัดศูนย์ท้ายเพื่อลบ padding
ถ้า ASM_Decoder ไม่คืน frame กราฟจะแสดงข้อความ ไม่มีการนำต้นฉบับมาแทนข้อมูลที่หาย

การถอดรหัสทำครบทุก block ก่อนตัดช่วงแสดงภาพ
แถว 1 และ 6 ใช้ payload bit index ส่วนแถว 2–5 ใช้ slot index ของ block ที่เลือก
สองชนิดแกนนี้ไม่ได้จับคู่กันแบบหนึ่งต่อหนึ่ง เนื่องจากการเข้ารหัสและ interleaving
การเห็นเส้นตรงกันในช่วงกราฟไม่ได้ยืนยันว่าข้อมูลส่วนที่เหลือถูกทั้งหมด

ไฟล์ใหม่มีคอมเมนต์ไทยตามหลักอ่านง่ายของ karpathy-guidelines
ไม่ได้แก้ Step 1–7 หรือไฟล์ EncoderDecoder_FirstPart ต้นฉบับ
