"""Step 3: สร้างและแสดง final output จากข้อมูลของ Step 1."""

import contextlib
import io
import runpy
import sys

from step02_scppm_encoder import encode_scppm


# ทำให้ข้อความภาษาไทยแสดงได้บน Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# โหลดตัวแปร data จากไฟล์ส่วนแรก
# ซ่อน print(data) ที่อยู่ท้ายไฟล์เดิม เพื่อให้เห็นเฉพาะ final output
with contextlib.redirect_stdout(io.StringIO()):
    first_part = runpy.run_path("step01_input_preparation.py")

input_blocks = first_part["data"]
ppm_blocks = []

for block in input_blocks:
    symbols = encode_scppm(block)
    ppm_blocks.append(symbols)

# บันทึกผลครบทุก symbol เป็นไฟล์ข้อความ อ่านได้ด้วย Notepad
with open("ppm_output.txt", "w", encoding="utf-8") as output_file:
    for block_number, symbols in enumerate(ppm_blocks, start=1):
        output_file.write(f"Block {block_number}\n")
        output_file.write(f"Number of PPM symbols: {len(symbols)}\n")

        # แบ่งบรรทัดละ 32 symbols เพื่อให้อ่านง่าย
        for start in range(0, len(symbols), 32):
            row = symbols[start : start + 32]
            output_file.write(" ".join(str(int(symbol)) for symbol in row))
            output_file.write("\n")

        output_file.write("\n")

print("สร้าง ppm_output.txt เรียบร้อย")
print("จำนวน blocks:", len(ppm_blocks))

for block_number, symbols in enumerate(ppm_blocks, start=1):
    print()
    print("Block:", block_number)
    print("จำนวน PPM symbols:", len(symbols))
    print("32 symbols แรก:", symbols[:32])
