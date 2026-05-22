# Rajah 3.1: Proses Kajian UMS

## 📋 Cara 2: Render dengan Tahoma Sebenar

Script ini memungkinkan anda untuk menghasilkan diagram profesional UMS dengan font Tahoma yang sesuai.

### ✅ Spesifikasi Output
- **Ukuran**: 1280 × 720 piksel (resolusi standard)
- **Warna**: Hitam teks, putih background (UMS-compliant)
- **Font**: Tahoma (atau fallback sans-serif jika tidak tersedia)
- **Format**: PNG (siap untuk Word, PowerPoint, atau cetak)

---

## 🚀 Cara Menggunakan

### Di **Windows** (Rekomendasi - untuk Tahoma sebenar):

1. **Pastikan Python sudah dipasang**
   ```bash
   python --version
   ```

2. **Pasang matplotlib** (jika belum):
   ```bash
   pip install matplotlib
   ```

3. **Jalankan script**:
   ```bash
   python build_rajah_3_1_proses_kajian.py
   ```

4. **Hasil**: Fail `rajah_3_1_proses_kajian.png` akan dijana di folder yang sama.

---

### Di **Linux/Mac** (Font akan fallback):

Sama seperti Windows, tetapi font akan gunakan Liberation Sans (kerana Tahoma tidak standard di Linux/Mac).

```bash
python build_rajah_3_1_proses_kajian.py
```

---

## 📊 Struktur Rajah

### **FASA 1: Perancangan Kajian** (5 langkah)
```
Mula
  ↓
Memahami konteks kajian
  ↓
Mendefinisikan masalah kajian
  ↓
Sorotan literatur & mengenal pasti jurang
  ↓
Membangunkan model kajian
  ↓
Membentuk persoalan, objektif dan hipotesis
```

### **FASA 2: Pelaksanaan Kajian** (7 langkah)
```
Merancang instrumen kajian
  ↓
Menjalankan ujian rintis
  ↓
Memilih sampel dan peserta kajian
  ↓
Mengumpul data melalui soal selidik
  ↓
Memeriksa dan menyaring data
  ↓
Analisis deskriptif data
  ↓
Pengujian anggapan model
```

### **FASA 3: Penyelesaian Kajian** (4 langkah)
```
Saringan data menggunakan SPSS v29
  ↓
Pengujian hipotesis melalui PLS-SEM dalam SmartPLS
  ↓
Tafsiran dapatan
  ↓
Perbincangan, rumusan dan pendokumentasian
  ↓
Tamat
```

---

## 💾 Cara Guna Rajah

### 1️⃣ **Dalam Word** (Paling mudah):
```
Insert > Pictures > This Device
→ Pilih rajah_3_1_proses_kajian.png
→ Tekan Insert
```

### 2️⃣ **Tambah Caption di Word**:
```
Klik rajah → Insert > Captions
Caption: "Rajah 3.1: Proses Kajian"
Label position: Below
```

### 3️⃣ **Dalam PowerPoint**:
```
Insert > Pictures > This Device
→ Pilih PNG
→ Resize mengikut slide
```

### 4️⃣ **Cetak ke A4**:
```
Saiz A4: 210 × 297 mm
Resolusi: 1280×720 piksel ✓ (sesuai untuk cetak)
Warna: Hitam-putih (ekonomi tinta)
```

---

## 🔧 Mengubah Rajah

Jika anda perlu mengubah teks atau struktur:

1. **Edit `build_rajah_3_1_proses_kajian.py`**
   - Cari fungsi `add_box()` dan `add_arrow()`
   - Ubah teks atau posisi

2. **Jalankan ulang script**:
   ```bash
   python build_rajah_3_1_proses_kajian.py
   ```

3. **PNG akan dikemas kini secara automatik**

---

## 📝 Nota

| Aspek | Windows | Linux/Mac |
|-------|---------|-----------|
| **Font** | Tahoma (sebenar) ✓ | Liberation Sans (fallback) |
| **Rendering** | Sempurna | Hampir sama |
| **Gaya** | UMS-compliant ✓ | UMS-compliant ✓ |
| **Cetak** | Tajam & jernih | Tajam & jernih |

**Cadangan**: Untuk hasil terbaik (Tahoma exact), jalankan script di **Windows**.

---

## 🆘 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'matplotlib'"
**Penyelesaian**:
```bash
pip install matplotlib
```

### ❌ Font tidak Tahoma
**Sebab**: Linux/Mac tidak ada Tahoma standard
**Penyelesaian**: Jalankan script di Windows, atau pasang Tahoma font di sistem anda

### ❌ PNG tidak dijana
**Penyelesaian**:
```bash
# Periksa folder semasa
pwd
# Sepatutnya ada rajah_3_1_proses_kajian.png di sini
ls -la
```

---

## 📧 Soalan?

Jika ada isu atau perlu ubah rajah, hubungi:
- **Script Author**: Claude Code
- **Sumber**: examples/ums_thesis/build_rajah_3_1_proses_kajian.py

---

**Selamat menggunakan Rajah 3.1 untuk tesis UMS anda!** 🎓
