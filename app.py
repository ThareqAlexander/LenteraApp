"""
LENTERA — Leprosy Early Recognition and Assessment
=====================================================
Web app deteksi dini kusta dengan tampilan mobile-style,
fitur scan kamera langsung + upload gambar, riwayat, dan info edukasi.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import json
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import efficientnet

# =========================================================
# KONFIGURASI — SESUAIKAN BAGIAN INI
# =========================================================
ZIP_PATH = "Model/MODEL.OLD.zip"
EXTRACT_PATH = "Model/extracted"
RIWAYAT_PATH = "riwayat.json"

CLASS_NAMES = ["Bukan kusta", "Kusta"]
IMG_SIZE = (300, 300)
THRESHOLD = 70  # dalam persen

USER_NAME = "Budi Santoso"  # ganti sesuai kebutuhan demo

st.set_page_config(page_title="Lentera", page_icon="🩺", layout="centered")

# =========================================================
# CSS — supaya tampilan menyerupai aplikasi mobile
# =========================================================
st.markdown("""
<style>
    .block-container {
    max-width: 430px;
    padding-top: 2.5rem !important;
    padding-bottom: 6rem;
    margin: auto;
}
header[data-testid="stHeader"] {
    height: 0rem;
}

div[data-testid="stAppViewContainer"] {
    padding-top: 0rem;
}
    .lentera-header {
        background: linear-gradient(135deg, #3b5fe2, #5b7cf0);
        border-radius: 18px;
        padding: 20px;
        color: white;
        margin-bottom: 16px;
    }
    .lentera-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        color: #1f2937;
    }
    .lentera-card p, .lentera-card li, .lentera-card b, .lentera-card span:not([class^="badge"]) {
        color: #1f2937;
    }
    .lentera-card ul { margin: 6px 0 0 0; padding-left: 18px; }
    .lentera-section-label {
        font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
        color: #9ca3af; margin: 4px 0 6px 4px; text-transform: uppercase;
    }
    .lentera-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 4px; border-bottom: 1px solid #f0f1f4;
    }
    .lentera-row:last-child { border-bottom: none; }
    .badge-rendah { background:#e7f8ee; color:#1e9e5a; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-sedang { background:#fff5e6; color:#d98a1a; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-tinggi { background:#fdeaea; color:#d63d3d; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-dark { background:#1f2937; color:#ffffff; padding:4px 10px; border-radius:20px; font-weight:600; font-size:12px; }
    div[data-testid="stBottomBlockContainer"] { max-width: 430px; margin: auto; }

    /* Tombol jangan wrap ke baris baru, teks disusutkan biar muat */
    div[data-testid="stButton"] button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 12px;
        padding: 8px 2px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource(show_spinner="Memuat model, mohon tunggu...")
def load_model():
    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    with open(os.path.join(EXTRACT_PATH, "config.json"), "r") as f:
        config = json.load(f)
    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(os.path.join(EXTRACT_PATH, "model.weights.h5"))
    return model


def predict_image(pil_img, model):
    img_resized = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img_resized)
    img_preprocessed = efficientnet.preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    return CLASS_NAMES[top_idx], float(predictions[top_idx] * 100)


# =========================================================
# VALIDASI FOTO — filter di luar model, sebelum prediksi
# =========================================================
MIN_SKIN_RATIO = 0.12  # minimal 12% piksel harus terdeteksi sebagai warna kulit

_face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def hitung_rasio_kulit(cv_img):
    """Menghitung persentase piksel yang masuk rentang warna kulit manusia (YCrCb)."""
    ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.sum(mask > 0)) / float(mask.size)


def validasi_foto(pil_img):
    """
    Mengembalikan (is_valid: bool, alasan: str).
    Menolak foto yang mengandung wajah, atau yang rasio warna kulitnya terlalu rendah.
    """
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return False, "Terdeteksi wajah pada foto. Mohon unggah foto close-up area kulit yang ingin diperiksa, bukan foto wajah/selfie."

    skin_ratio = hitung_rasio_kulit(cv_img)
    if skin_ratio < MIN_SKIN_RATIO:
        return False, "Foto tidak menunjukkan area kulit yang jelas. Coba ambil foto lebih dekat dan pastikan pencahayaan cukup."

    return True, ""


def status_badge(pred_class):
    """Menampilkan status klasifikasi apa adanya (bukan penilaian tingkat risiko klinis)."""
    if pred_class == "Bukan kusta":
        return "Tidak Terdeteksi", "badge-rendah"
    return "Kusta Terdeteksi", "badge-tinggi"


def keyakinan_tier(pred_conf):
    """Kategori tingkat keyakinan model (bukan penilaian risiko klinis)."""
    if pred_conf >= THRESHOLD:
        return "Keyakinan Tinggi", "badge-tinggi"
    if pred_conf >= 50:
        return "Keyakinan Sedang", "badge-sedang"
    return "Keyakinan Rendah", "badge-rendah"


def rekomendasi_text(pred_class, pred_conf, input_valid=True):
    if not input_valid:
        return (
            "Foto tidak dapat dianalisis karena bukan merupakan foto area kulit "
            "yang sesuai. Silakan unggah foto close-up bagian kulit yang ingin diperiksa."
        )

    if pred_class == "Bukan kusta":
        return (
            "Tidak ditemukan tanda-tanda kusta yang signifikan pada foto yang dianalisis. "
            "Tetap jaga kebersihan kulit dan periksa ke fasilitas kesehatan jika muncul "
            "perubahan pada kulit."
        )

    if pred_conf >= THRESHOLD:
        return (
            "Terdapat indikasi yang perlu diperhatikan. Segera kunjungi Puskesmas "
            "atau fasilitas kesehatan terdekat untuk pemeriksaan lebih lanjut."
        )

    return (
        "Hasil menunjukkan indikasi yang belum meyakinkan. Pantau kondisi kulit "
        "dan pertimbangkan pemeriksaan ke fasilitas kesehatan jika terdapat perubahan."
    )
    def buat_hasil_invalid(filename, alasan):
    return {
        "filename": filename,
        "class": "Input Foto Tidak Valid",
        "confidence": 0,
        "label": "Input Tidak Valid",
        "badge_class": "badge-sedang",
        "tier_label": "Tidak Dapat Dianalisis",
        "tier_class": "badge-sedang",
        "rekomendasi": (
            "Foto tidak dapat dianalisis karena bukan merupakan foto area kulit "
            "yang sesuai. Silakan unggah foto close-up area kulit yang ingin diperiksa."
        ),
        "alasan": alasan,
        "date": datetime.now().strftime("%d %b %Y, %H:%M"),
        "input_valid": False
    }


# =========================================================
# RIWAYAT (baca/tulis riwayat.json)
# =========================================================
def load_riwayat():
    if os.path.exists(RIWAYAT_PATH):
        try:
            with open(RIWAYAT_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_riwayat(entry):
    data = load_riwayat()
    data.insert(0, entry)
    with open(RIWAYAT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# STATE NAVIGASI
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None


def go_to(page_name):
    st.session_state.page = page_name


# =========================================================
# HALAMAN: BERANDA
# =========================================================
def halaman_beranda():
    st.markdown(f"""
    <div class="lentera-header">
        <h2 style="margin:0;">🌿 LENTERA</h2>
        <p style="margin:4px 0 0 0;">Selamat datang kembali,</p>
        <h3 style="margin:0;">{USER_NAME} 👋</h3>
        <p style="opacity:0.85; margin-top:4px;">Leprosy Early Recognition and Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📷\nScan", use_container_width=True):
            go_to("Scan")
            st.rerun()
    with c2:
        if st.button("📍\nFaskes", use_container_width=True):
            go_to("Faskes")
            st.rerun()
    with c3:
        if st.button("🕘\nRiwayat", use_container_width=True):
            go_to("Riwayat")
            st.rerun()

    st.markdown("""
    <div class="lentera-card">
        <b>📖 Apa itu Kusta?</b>
        <p style="margin-top:8px;">Penyakit infeksi kronis akibat bakteri <i>Mycobacterium leprae</i>
        yang menyerang kulit dan saraf tepi.</p>
        <ul>
            <li>Penularan melalui kontak erat jangka panjang</li>
            <li>Tanda awal: bercak pucat/mati rasa pada kulit</li>
            <li>Dapat disembuhkan dengan MDT (Multi Drug Therapy) sejak dini</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HALAMAN: SCAN (kamera langsung + upload)
# =========================================================
def halaman_scan():
    st.markdown("### 📷 Scan Kulit")
    st.caption("Deteksi tanda-tanda kusta dengan AI")

    if "scan_mode" not in st.session_state:
        st.session_state.scan_mode = "Buka Kamera"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📷 Buka Kamera", use_container_width=True,
                      type="primary" if st.session_state.scan_mode == "Buka Kamera" else "secondary"):
            st.session_state.scan_mode = "Buka Kamera"
            st.rerun()
    with c2:
        if st.button("🖼️ Dari Galeri", use_container_width=True,
                      type="primary" if st.session_state.scan_mode == "Dari Galeri" else "secondary"):
            st.session_state.scan_mode = "Dari Galeri"
            st.rerun()

    img_input = None
    filename = None

    if st.session_state.scan_mode == "Buka Kamera":
        cam_result = st.camera_input("Arahkan kamera ke area kulit yang ingin diperiksa", label_visibility="collapsed")
        if cam_result is not None:
            img_input = Image.open(cam_result)
            filename = f"scan_kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    else:
        uploaded_file = st.file_uploader("Pilih gambar dari galeri", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file)
            filename = uploaded_file.name

    if img_input is not None:
        st.image(img_input, use_container_width=True)

       if st.button(
    "🔍 Analisis Gambar",
    use_container_width=True,
    type="primary"
):

    # =====================================================
    # 1. VALIDASI FOTO SEBELUM MASUK MODEL
    # =====================================================
    with st.spinner("Memeriksa validitas foto..."):
        is_valid, alasan = validasi_foto(img_input)

    if not is_valid:

        # Foto tidak valid
        st.session_state.scan_result = buat_hasil_invalid(
            filename,
            alasan
        )

        save_riwayat(st.session_state.scan_result)

        st.warning("⚠️ Input Foto Tidak Valid")
        st.info(alasan)

    else:

        # =================================================
        # 2. FOTO VALID → LANJUT KE MODEL AI
        # =================================================
        with st.spinner("Menganalisis kondisi kulit..."):
            model = load_model()
            pred_class, pred_conf = predict_image(
                img_input,
                model
            )

        label, badge_class = status_badge(pred_class)
        tier_label, tier_class = keyakinan_tier(pred_conf)
        rekomendasi = rekomendasi_text(
            pred_class,
            pred_conf,
            input_valid=True
        )

        st.session_state.scan_result = {
            "filename": filename,
            "class": pred_class,
            "confidence": round(pred_conf, 2),
            "label": label,
            "badge_class": badge_class,
            "tier_label": tier_label,
            "tier_class": tier_class,
            "rekomendasi": rekomendasi,
            "date": datetime.now().strftime(
                "%d %b %Y, %H:%M"
            ),
            "input_valid": True
        }

        save_riwayat(
            st.session_state.scan_result
        )

            st.session_state.scan_result = {
                "filename": filename,
                "class": pred_class,
                "confidence": round(pred_conf, 2),
                "label": label,
                "badge_class": badge_class,
                "tier_label": tier_label,
                "tier_class": tier_class,
                "rekomendasi": rekomendasi,
                "date": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            save_riwayat(st.session_state.scan_result)

   if st.session_state.scan_result:

    r = st.session_state.scan_result

    # =====================================================
    # HASIL INPUT TIDAK VALID
    # =====================================================
    if not r.get("input_valid", True):

        st.markdown(f"""
        <div class="lentera-card"
             style="background:#fff8e8; border-color:#f5d58a;">

            <span class="badge-sedang">
                ⚠️ Input Tidak Valid
            </span>

            <p style="margin-top:14px;">
                <b>Hasil Pemeriksaan</b>
            </p>

            <p style="font-size:18px; font-weight:700;">
                📷 Bukan Foto Kulit
            </p>

            <p style="margin-top:6px;">
                {r.get("alasan", "Foto tidak sesuai untuk dianalisis.")}
            </p>

        </div>

        <div class="lentera-card"
             style="background:#eef5ff; border-color:#d5e5ff;">

            <b>💡 Rekomendasi</b>

            <p style="margin-top:6px;">
                Silakan ambil foto <b>close-up area kulit</b> yang ingin
                diperiksa. Hindari foto wajah/selfie, benda, pemandangan,
                atau gambar yang tidak memperlihatkan kondisi kulit secara jelas.
            </p>

        </div>

        <p style="font-size:12px; color:#9ca3af;">
            ⚠️ Sistem hanya dapat melakukan analisis pada foto area kulit
            yang sesuai dengan kebutuhan pemeriksaan.
        </p>
        """, unsafe_allow_html=True)

    # =====================================================
    # HASIL MODEL
    # =====================================================
    else:

        tier_class = r.get(
            "tier_class",
            "badge-rendah"
        )

        tier_label = r.get(
            "tier_label",
            "-"
        )

        st.markdown(f"""
        <div class="lentera-card">

            <span class="{tier_class}">
                {tier_label}
            </span>

            <span class="badge-dark"
                  style="float:right;">
                {r['label']}
            </span>

            <p style="margin-top:14px;">
                <b>Kelas: {r['class']}</b>
            </p>

            <p style="margin-top:2px;">
                <b>Tingkat Keyakinan Model</b>
            </p>

            <div style="
                background:#eee;
                border-radius:8px;
                height:10px;
                margin-bottom:4px;
            ">

                <div style="
                    background:#2ecc71;
                    width:{r['confidence']}%;
                    height:10px;
                    border-radius:8px;
                "></div>

            </div>

            <p style="
                text-align:right;
                font-size:13px;
                color:#666;
            ">
                {r['confidence']}%
            </p>

        </div>

        <div class="lentera-card"
             style="
                background:#eefbf1;
                border-color:#d4f3e0;
             ">

            <b>✅ Rekomendasi</b>

            <p style="margin-top:6px;">
                {r['rekomendasi']}
            </p>

        </div>

        <p style="
            font-size:12px;
            color:#9ca3af;
        ">
            ⚠️ Hasil ini adalah estimasi berbasis AI,
            bukan diagnosis medis. Selalu konsultasikan
            ke tenaga kesehatan untuk kepastian.
        </p>

        """, unsafe_allow_html=True)
        tier_class = r.get("tier_class", "badge-rendah")
        tier_label = r.get("tier_label", "-")
        st.markdown(f"""
        <div class="lentera-card">
            <span class="{tier_class}">{tier_label}</span>
            <span class="badge-dark" style="float:right;">{r['label']}</span>
            <p style="margin-top:14px;"><b>Kelas: {r['class']}</b></p>
            <p style="margin-top:2px;"><b>Tingkat Keyakinan Model</b></p>
            <div style="background:#eee; border-radius:8px; height:10px; margin-bottom:4px;">
                <div style="background:#2ecc71; width:{r['confidence']}%; height:10px; border-radius:8px;"></div>
            </div>
            <p style="text-align:right; font-size:13px; color:#666;">{r['confidence']}%</p>
        </div>
        <div class="lentera-card" style="background:#eefbf1; border-color:#d4f3e0;">
            <b>✅ Rekomendasi</b>
            <p style="margin-top:6px;">{r['rekomendasi']}</p>
        </div>
        <p style="font-size:12px; color:#9ca3af;">⚠️ Hasil ini adalah estimasi berbasis AI, bukan diagnosis medis. Selalu konsultasikan ke tenaga kesehatan untuk kepastian.</p>
        """, unsafe_allow_html=True)


# =========================================================
# HALAMAN: FASKES (data contoh — belum terhubung API lokasi asli)
# =========================================================
def halaman_faskes():
    st.markdown("### 📍 Faskes Terdekat")
    st.caption("Berdasarkan lokasi Anda saat ini")
    st.text_input("🔍 Cari fasilitas kesehatan...")

    faskes_dummy = [
        {"nama": "RSUD Dr. H. Moh. Anwar", "tipe": "Rumah Sakit Umum", "jarak": "1.2 km", "status": "Buka"},
        {"nama": "Puskesmas Kota Sumenep", "tipe": "Puskesmas", "jarak": "0.8 km", "status": "Buka"},
        {"nama": "Klinik Pratama Sehat Sejahtera", "tipe": "Klinik Pratama", "jarak": "2.3 km", "status": "Tutup"},
    ]
    st.caption(f"{len(faskes_dummy)} FASKES DITEMUKAN (data contoh)")
    for i, f in enumerate(faskes_dummy):
        warna = "#1e9e5a" if f["status"] == "Buka" else "#d63d3d"
        st.markdown(f"""
        <div class="lentera-card">
            <b>{f['nama']}</b><br>
            <span style="color:#666; font-size:13px;">{f['tipe']} · {f['jarak']}</span><br>
            <span style="color:{warna}; font-size:13px;">● {f['status']}</span>
        </div>
        """, unsafe_allow_html=True)
        if i == 1:
            st.button("🎫 Ambil Nomor Antrian", key=f"antrian_{i}", use_container_width=True)
    st.caption("Catatan: data faskes di atas masih contoh statis. Untuk lokasi real-time dibutuhkan integrasi API peta (mis. Google Places).")


# =========================================================
# HALAMAN: RIWAYAT
# =========================================================
def halaman_riwayat():

    st.markdown("### 🕘 Riwayat Pemeriksaan")

    data = load_riwayat()

    # =====================================================
    # HEADER RIWAYAT
    # =====================================================

    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption(
            f"{len(data)} pemeriksaan tersimpan"
        )

    with col2:
        if data:
            if st.button(
                "🧹 Bersihkan",
                use_container_width=True
            ):
                st.session_state.confirm_clear = True

    # =====================================================
    # KONFIRMASI HAPUS SEMUA
    # =====================================================

    if st.session_state.get(
        "confirm_clear",
        False
    ):

        st.warning(
            "Apakah kamu yakin ingin menghapus "
            "seluruh riwayat pemeriksaan?"
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "Ya, Hapus Semua",
                use_container_width=True,
                type="primary"
            ):

                clear_riwayat()

                st.session_state.confirm_clear = False
                st.session_state.scan_result = None

                st.success(
                    "Seluruh riwayat berhasil dihapus."
                )

                st.rerun()

        with c2:
            if st.button(
                "Batal",
                use_container_width=True
            ):

                st.session_state.confirm_clear = False
                st.rerun()

    # =====================================================
    # JIKA BELUM ADA RIWAYAT
    # =====================================================

    if not data:

        st.info(
            "Belum ada riwayat pemeriksaan. "
            "Coba lakukan scan terlebih dahulu."
        )

        return

    # =====================================================
    # DAFTAR RIWAYAT
    # =====================================================

    for i, entry in enumerate(data):

        tier_label = entry.get(
            "tier_label",
            entry.get("label", "-")
        )

        tier_class = entry.get(
            "tier_class",
            entry.get("badge_class", "badge-rendah")
        )

        input_valid = entry.get(
            "input_valid",
            True
        )

        # Tentukan tampilan untuk input invalid
        if not input_valid:

            status_text = "Input Tidak Valid"
            status_class = "badge-sedang"

        else:

            status_text = tier_label
            status_class = tier_class

        st.markdown(
            f"""
            <div class="lentera-card">

                <b>
                    {entry.get('filename', '-')}
                </b>

                <span class="{status_class}"
                      style="float:right;">
                    {status_text}
                </span>

                <br>

                <span style="
                    color:#888;
                    font-size:12px;
                ">
                    {entry.get('date', '-')}
                </span>

                <br>

                <span style="
                    color:#666;
                    font-size:13px;
                ">
                    {entry.get('class', '-')}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        # Tombol hapus individual
        if st.button(
            "🗑️ Hapus",
            key=f"hapus_riwayat_{i}",
            use_container_width=True
        ):

            delete_riwayat(i)

            st.success(
                "Riwayat berhasil dihapus."
            )

            st.rerun()


# =========================================================
# HALAMAN: AKUN
# =========================================================
def halaman_akun():
    st.markdown("### 👤 Akun Saya")
    st.markdown(f"""
    <div class="lentera-header" style="display:flex; align-items:center; gap:14px;">
        <div style="font-size:36px;">🧑</div>
        <div>
            <b style="font-size:16px;">{USER_NAME}</b><br>
            <span style="opacity:0.85; font-size:13px;">budi.santoso@email.com</span><br>
            <span style="opacity:0.85; font-size:13px;">NIK: •••••••••053492</span>
        </div>
    </div>

    <div class="lentera-section-label">Profil</div>
    <div class="lentera-card">
        <div class="lentera-row"><span>✏️ Edit Profil</span><span>›</span></div>
    </div>

    <div class="lentera-section-label">Preferensi</div>
    <div class="lentera-card">
        <div class="lentera-row"><span>🔔 Pengaturan Notifikasi</span><span>›</span></div>
        <div class="lentera-row"><span>🌐 Bahasa</span><span>Indonesia ›</span></div>
    </div>

    <div class="lentera-section-label">Privasi & Keamanan</div>
    <div class="lentera-card">
        <div class="lentera-row"><span>🔒 Privasi & Data</span><span>›</span></div>
        <div class="lentera-row"><span>💳 Koneksikan dengan JKN</span><span style="color:#3b5fe2;">Hubungkan</span></div>
    </div>
    <p style="font-size:12px; color:#9ca3af;">Menghubungkan JKN memungkinkan data skrining Anda terkoordinasi dengan layanan BPJS Kesehatan.</p>
    """, unsafe_allow_html=True)


# =========================================================
# RENDER HALAMAN AKTIF
# =========================================================
pages = {
    "Beranda": halaman_beranda,
    "Faskes": halaman_faskes,
    "Scan": halaman_scan,
    "Riwayat": halaman_riwayat,
    "Akun": halaman_akun,
}
pages[st.session_state.page]()

# =========================================================
# BOTTOM NAVIGATION
# =========================================================
st.write("")
nav_labels = ["🏠 Beranda", "📍 Faskes", "📷 Scan", "🕘 Riwayat", "👤 Akun"]
nav_keys = ["Beranda", "Faskes", "Scan", "Riwayat", "Akun"]
cols = st.columns(5)
for col, label, key in zip(cols, nav_labels, nav_keys):
    with col:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            go_to(key)
            st.rerun()
