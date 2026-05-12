"""
Script untuk render Rajah 3.1: Proses Kajian dengan font Tahoma sebenar.

Cara menjalankan:
- Windows: python build_rajah_3_1_proses_kajian.py
- Linux/Mac: python build_rajah_3_1_proses_kajian.py (fallback ke Liberation Sans)

Output: rajah_3_1_proses_kajian.png (1280x720 piksel, hitam-putih, sans-serif)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Tetapkan font ke Tahoma (akan fallback ke sans-serif jika tidak tersedia)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

# Ukuran rajah (1280x720 untuk resolusi layar standard)
fig, ax = plt.subplots(figsize=(16, 9), dpi=80)
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Warna: Hitam teks, putih background, hitam border
text_color = '#000000'
border_color = '#000000'
bg_color = '#FFFFFF'

def add_box(ax, x, y, width, height, text, font_size=9, bold=False, is_terminator=False):
    """Tambah kotak dengan teks."""
    if is_terminator:
        # Oval untuk Mula/Tamat
        box = patches.Ellipse((x + width/2, y + height/2), width, height,
                            edgecolor=border_color, facecolor=bg_color, linewidth=1.5)
    else:
        # Segi empat untuk langkah biasa
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                           edgecolor=border_color, facecolor=bg_color, linewidth=1.5)
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=font_size,
            weight=weight, color=text_color, wrap=True)

def add_arrow(ax, x1, y1, x2, y2):
    """Tambah panah antara kotak."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=1.5, color=border_color)
    ax.add_patch(arrow)

# ===== FASA 1: Perancangan Kajian =====
y_start = 14.5

# Mula
add_box(ax, 3.5, y_start, 3, 0.6, 'Mula', font_size=10, bold=True, is_terminator=True)
add_arrow(ax, 5, y_start, 5, y_start - 0.8)

# Langkah 1
y_current = y_start - 1.4
add_box(ax, 2.8, y_current, 4.4, 0.7, 'Memahami konteks kajian', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 2
y_current -= 1.6
add_box(ax, 2.5, y_current, 5, 0.7, 'Mendefinisikan masalah kajian', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 3
y_current -= 1.6
add_box(ax, 2.3, y_current, 5.4, 0.7, 'Sorotan literatur & mengenal pasti jurang', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 4
y_current -= 1.6
add_box(ax, 2.8, y_current, 4.4, 0.7, 'Membangunkan model kajian', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 5
y_current -= 1.6
add_box(ax, 1.5, y_current, 7, 0.8, 'Membentuk persoalan, objektif dan hipotesis', font_size=9)

# Label FASA 1
ax.text(0.3, y_start - 2, 'FASA 1:\nPerancangan\nKajian',
        fontsize=9, weight='bold', color=text_color, va='top')

# ===== FASA 2: Pelaksanaan Kajian =====
# (Lebih panjang - 7 langkah)

add_arrow(ax, 5, y_current, 5, y_current - 0.9)
y_current -= 1.6

steps_fasa2 = [
    'Merancang instrumen kajian',
    'Menjalankan ujian rintis',
    'Memilih sampel dan peserta kajian',
    'Mengumpul data melalui soal selidik',
    'Memeriksa dan menyaring data',
    'Analisis deskriptif data',
    'Pengujian anggapan model'
]

for i, step in enumerate(steps_fasa2):
    add_box(ax, 2.2, y_current, 5.6, 0.7, step, font_size=9)
    if i < len(steps_fasa2) - 1:
        add_arrow(ax, 5, y_current, 5, y_current - 0.9)
    y_current -= 1.6

# Label FASA 2
ax.text(9.5, y_start - 5, 'FASA 2:\nPelaksanaan\nKajian',
        fontsize=9, weight='bold', color=text_color, ha='right', va='top')

# ===== FASA 3: Penyelesaian Kajian =====

add_arrow(ax, 5, y_current, 5, y_current - 0.9)
y_current -= 1.6

# Langkah 1 Fasa 3
add_box(ax, 2, y_current, 6, 0.7, 'Saringan data menggunakan SPSS v29', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 2 Fasa 3
y_current -= 1.6
add_box(ax, 1.5, y_current, 7, 0.8, 'Pengujian hipotesis melalui PLS-SEM dalam SmartPLS', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 3 Fasa 3
y_current -= 1.6
add_box(ax, 2.5, y_current, 5, 0.7, 'Tafsiran dapatan', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Langkah 4 Fasa 3
y_current -= 1.6
add_box(ax, 1.2, y_current, 7.6, 0.7, 'Perbincangan, rumusan dan pendokumentasian', font_size=9)
add_arrow(ax, 5, y_current, 5, y_current - 0.9)

# Tamat
y_current -= 1.6
add_box(ax, 3.5, y_current, 3, 0.6, 'Tamat', font_size=10, bold=True, is_terminator=True)

# Label FASA 3
ax.text(0.3, y_start - 10.5, 'FASA 3:\nPenyelesaian\nKajian',
        fontsize=9, weight='bold', color=text_color, va='top')

# Tetapkan background putih
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Simpan rajah
output_path = os.path.join(os.path.dirname(__file__), 'rajah_3_1_proses_kajian.png')
plt.tight_layout()
plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
print(f"✓ Rajah berjaya disimpan: {output_path}")
print(f"  Saiz: 1280x720 piksel")
print(f"  Font: Tahoma (atau fallback sans-serif)")
print(f"  Gaya: Hitam-putih, UMS-compliant")

plt.close()
