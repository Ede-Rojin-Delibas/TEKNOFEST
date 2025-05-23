from flask import request, jsonify
from app.api import bp
from app.utils.net_hesapla import net_hesapla
from app.utils.net_hesapla import toplam_net_hesapla
from app.utils.tahmin import tahmin_yap
from config import Config

# Geçerli sınav türlerini config'den al
GECERLI_SINAV_TURLERI = Config.ALLOWED_EXAM_TYPES

@bp.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running"})

@bp.route('/tahmin', methods=['POST'])
def tahmin_et():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Veri gönderilmedi"}), 400

    # 1. Sınav türü kontrolü
    sinav_turu = data.get("sinav_turu", "").lower()
    if sinav_turu not in GECERLI_SINAV_TURLERI:
        return jsonify({
            "error": "Geçersiz sınav türü",
            "gecerli_turler": GECERLI_SINAV_TURLERI
        }), 400

    # 2. Hedef sıralama kontrolü - sadece pozitif sayı olması yeterli
    try:
        hedef_siralama = int(data.get("hedef_siralama", 0))
        if hedef_siralama <= 0:
            return jsonify({"error": "Hedef sıralama pozitif bir sayı olmalıdır"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Hedef sıralama sayısal bir değer olmalıdır"}), 400

    # 3. Doğru/Yanlış verileri kontrolü
    dogru_yanlis_verileri = data.get("dogru_yanlis")
    if not isinstance(dogru_yanlis_verileri, dict) or not dogru_yanlis_verileri:
        return jsonify({"error": "Doğru/yanlış verileri eksik ya da hatalı formatta"}), 400

    # 4. Net Hesaplama
    try:
        netler = {}
        for ders, dy in dogru_yanlis_verileri.items():
            netler[ders] = net_hesapla(sinav_turu, dy)
        toplam_net = toplam_net_hesapla(sinav_turu, netler)
    except Exception as e:
        return jsonify({"error": f"Net hesaplama hatası: {str(e)}"}), 500

    # 5. Sıralama Tahmini
    try:
        tahmini_siralama = tahmin_yap(sinav_turu, toplam_net)
    except Exception as e:
        return jsonify({"error": f"Tahmin hatası: {str(e)}"}), 500

    # 6. Sonuç Değerlendirmesi
    fark = abs(hedef_siralama - tahmini_siralama)
    durum = "Hedefinize çok yakınsınız!" if fark < 5000 else "Hedefinizle arada fark var, netlerinizi biraz artırmalısınız."

    # 7. JSON Cevap
    response = {
        "sinav_turu": sinav_turu,
        "net": {
            "ders_bazli": netler,
            "toplam": toplam_net
        },
        "siralama": {
            "tahmini": tahmini_siralama,
            "hedef": hedef_siralama,
            "fark": fark,
            "degerlendirme": durum
        }
    }
    
    return jsonify(response)
