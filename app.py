from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
const port = process.env.PORT || 5000

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FOLDER = os.path.join(BASE_DIR, 'xgboost_result')
MRL_DATA_PATH = os.path.join(BASE_DIR, 'fao_pesticide_mrl_dataset.csv')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'egitim_icin_tam_sentetik_veri_formul_v3.csv')

MODEL_PATH = os.path.join(RESULT_FOLDER, 'xgboost_model_result.pkl')
SCALER_PATH = os.path.join(RESULT_FOLDER, 'scaler.pkl')
ENCODERS_PATH = os.path.join(RESULT_FOLDER, 'encoders.pkl')

# Global Değişkenler
model = None
scaler = None
encoders = {}
mrl_lookup = {}
ppm_lookup = {}
vp_lookup = {}          # YENİ: Pestisit -> Buhar Basıncı
tutunma_lookup = {}
tutunma_urun_yedek = {}
product_map = {}

# --- 1. VERİLERİ YÜKLE ---

# Eğitim Verisi Analizi
if os.path.exists(TRAIN_DATA_PATH):
    try:
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        
        # 1. Saf Madde PPM Hafızası
        ppm_map = df_train[['Pestisit', '1ml_Saf_Madde_ppm']].drop_duplicates().set_index('Pestisit')
        ppm_lookup = ppm_map['1ml_Saf_Madde_ppm'].to_dict()

        # 2. YENİ: Buhar Basıncı Hafızası
        vp_map = df_train[['Pestisit', 'Buhar_Basinci_Ref_25C']].drop_duplicates().set_index('Pestisit')
        vp_lookup = vp_map['Buhar_Basinci_Ref_25C'].to_dict()
        
        # 3. Tutunma Katsayısı
        df_train['Hesaplanan_Tutunma'] = df_train['Tahmini_Kalinti_ppm'] / df_train['Uygulanan_Ilac_Miktari_ml']
        tutunma_lookup = df_train.groupby(['Pestisit', 'Urun'])['Hesaplanan_Tutunma'].mean().to_dict()
        tutunma_urun_yedek = df_train.groupby('Urun')['Hesaplanan_Tutunma'].mean().to_dict()
        
        # 4. Ürün Haritası (Validasyon için)
        valid_products = set(df_train['Urun'].unique())
        
        print(f"✅ Eğitim Verisi: PPM, VP ve Tutunma bilgileri öğrenildi.")
    except Exception as e:
        print(f"⚠️ Eğitim verisi hatası: {e}")

# MRL Verisi Analizi
if os.path.exists(MRL_DATA_PATH):
    try:
        df_mrl = pd.read_csv(MRL_DATA_PATH)
        df_mrl.columns = [c.strip() for c in df_mrl.columns]
        
        fao_grouped = df_mrl.groupby('pestisit_adi')['kullanildigi_urun'].apply(set).to_dict()
        
        for _, row in df_mrl.iterrows():
            p = str(row['pestisit_adi']).strip()
            u = str(row['kullanildigi_urun']).strip()
            try: mrl_lookup[(p, u)] = float(row['mrl_limiti'])
            except: continue
        
        # Dinamik Ürün Haritası
        if 'valid_products' in locals():
            for p, u_set in fao_grouped.items():
                p_clean = str(p).strip()
                valid_u_for_p = u_set.intersection(valid_products)
                if valid_u_for_p:
                    product_map[p_clean] = sorted(list(valid_u_for_p))
                    
        print(f"✅ MRL Verisi: Limitler ve Ürün Haritası hazır.")
    except Exception as e:
        print(f"⚠️ MRL verisi hatası: {e}")

# --- 2. MODEL BİLEŞENLERİ ---
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        print("✅ Yapay Zeka Modeli Hazır.")
    except Exception as e:
        print(f"❌ Model hatası: {e}")

# --- 3. CONFİG ---
FINAL_COLUMN_ORDER = [
    'Pestisit', 'Urun', 'MRL_Limiti', 'Buhar_Basinci_Ref_25C',
    'Sicaklik_C', 'Sicaklik_Sinifi', 'Ucuculuk_Gosterge_Index',
    'Ucuculuk_Sinifi', '1ml_Saf_Madde_ppm', 'Uygulanan_Ilac_Miktari_ml',
    'Tahmini_Kalinti_ppm'
]
KATEGORIK_SUTUNLAR = ['Pestisit', 'Urun'] 
SAYISAL_SUTUNLAR = [c for c in FINAL_COLUMN_ORDER if c not in ['Pestisit', 'Urun', 'Ucuculuk_Sinifi']]

@app.route('/')
def home():
    pestisit_listesi = sorted(list(product_map.keys())) if product_map else []
    if not pestisit_listesi and 'Pestisit' in encoders:
        pestisit_listesi = sorted(encoders['Pestisit'].classes_)

    return render_template('index.html', 
                           pestisitler=pestisit_listesi, 
                           product_map=product_map)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = {}
        hesaplanan_sinif_adi = ""

        gelen_pestisit = str(data.get('Pestisit')).strip()
        gelen_urun = str(data.get('Urun')).strip()

        # 1. OTOMATİK VERİLER
        # A. MRL
        mrl_degeri = mrl_lookup.get((gelen_pestisit, gelen_urun))
        if mrl_degeri is None: return jsonify({'error': f"❌ '{gelen_pestisit}' - '{gelen_urun}' için MRL bulunamadı!"}), 400
        input_data['MRL_Limiti'] = mrl_degeri

        # B. PPM
        saf_madde_ppm = ppm_lookup.get(gelen_pestisit)
        if saf_madde_ppm is None: return jsonify({'error': f"❌ '{gelen_pestisit}' için PPM değeri bulunamadı!"}), 400
        input_data['1ml_Saf_Madde_ppm'] = saf_madde_ppm

        # C. BUHAR BASINCI (OTOMATİK)
        buhar_basinci = vp_lookup.get(gelen_pestisit)
        if buhar_basinci is None: return jsonify({'error': f"❌ '{gelen_pestisit}' için Buhar Basıncı değeri bulunamadı!"}), 400
        input_data['Buhar_Basinci_Ref_25C'] = buhar_basinci

        # 2. KATEGORİK
        for col in KATEGORIK_SUTUNLAR:
            val = data.get(col)
            encoder = encoders.get(col)
            if encoder:
                try: input_data[col] = encoder.transform([str(val)])[0]
                except ValueError: return jsonify({'error': f"HATA: '{val}' eğitim setinde yok!"}), 400

        # 3. HESAPLAMALAR
        try:
            sicaklik_c = float(data.get('Sicaklik_C'))
            uygulanan_ml = float(data.get('Uygulanan_Ilac_Miktari_ml'))

            # Tutunma & Kalıntı
            tutunma_katsayisi = tutunma_lookup.get((gelen_pestisit, gelen_urun))
            if tutunma_katsayisi is None: tutunma_katsayisi = tutunma_urun_yedek.get(gelen_urun, 0.05)
            
            tahmini_kalinti = uygulanan_ml * tutunma_katsayisi
            tahmini_kalinti = round(tahmini_kalinti, 4)
            input_data['Tahmini_Kalinti_ppm'] = tahmini_kalinti

            # Sıcaklık Sınıfı
            sicaklik_sinifi = 1 if sicaklik_c > 25 else 0
            
            # Uçuculuk İndeksi
            sicaklik_kelvin = sicaklik_c + 273.15
            if buhar_basinci <= 0: return jsonify({'error': "Hatalı basınç değeri"}), 400
            ucuculuk_gosterge = np.log10(buhar_basinci) * (sicaklik_kelvin / 298.0)
            ucuculuk_gosterge = round(ucuculuk_gosterge, 4)

            # Uçuculuk Sınıfı
            if ucuculuk_gosterge > 2: hesaplanan_sinif_adi = "Gaz (Cok Yuksek)"
            elif ucuculuk_gosterge > 0: hesaplanan_sinif_adi = "Yuksek"
            elif ucuculuk_gosterge > -2: hesaplanan_sinif_adi = "Orta"
            elif ucuculuk_gosterge > -5: hesaplanan_sinif_adi = "Dusuk"
            else: hesaplanan_sinif_adi = "Cok Dusuk (Kalici)"
            
            if 'Ucuculuk_Sinifi' in encoders:
                input_data['Ucuculuk_Sinifi'] = encoders['Ucuculuk_Sinifi'].transform([hesaplanan_sinif_adi])[0]

            input_data['Sicaklik_C'] = sicaklik_c
            input_data['Sicaklik_Sinifi'] = sicaklik_sinifi
            input_data['Ucuculuk_Gosterge_Index'] = ucuculuk_gosterge
            input_data['Uygulanan_Ilac_Miktari_ml'] = uygulanan_ml

        except (TypeError, ValueError): return jsonify({'error': "Sayısal değerler hatalı!"}), 400

        # 4. TAHMİN
        df_input = pd.DataFrame([input_data])
        df_input = df_input[FINAL_COLUMN_ORDER]
        X_scaled = scaler.transform(df_input.values)
        prediction = model.predict(X_scaled)
        prob = model.predict_proba(X_scaled)[:, 1]
        
        guven_orani = round(float(prob[0]) * 100, 2)
        sonuc_mesaji = "RİSKLİ (Limit Üstü)" if prediction[0] == 1 else "GÜVENLİ (Limit Altı)"

        return jsonify({
            'prediction': int(prediction[0]),
            'result_text': sonuc_mesaji,
            'confidence': guven_orani,
            'found_mrl': mrl_degeri,
            'calc_index': ucuculuk_gosterge,
            'calc_kalinti': tahmini_kalinti,
            'auto_ppm': saf_madde_ppm,
            'auto_tutunma': round(tutunma_katsayisi, 4),
            'auto_vp': buhar_basinci # Kullanıcıya hangi basıncın kullanıldığını gösterelim
        })

    except Exception as e: return jsonify({'error': f"Sistem Hatası: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host=0.0.0.0, debug=True)
