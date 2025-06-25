from flask import request, jsonify, current_app
from app.api import bp
from app.utils.net_hesapla import net_hesapla
from app.utils.net_hesapla import toplam_net_hesapla
from app.utils.tahmin import tahmin_yap
from config import Config
import logging

logger = logging.getLogger(__name__)

# Geçerli sınav türlerini config'den al
GECERLI_SINAV_TURLERI = Config.ALLOWED_EXAM_TYPES

@bp.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running"})

@bp.route('/tahmin', methods=['POST'])
def tahmin_et():
    try:
        data = request.get_json()
        if not data:
            logger.error("Veri gönderilmedi")
            return jsonify({"error": "Veri gönderilmedi"}), 400

        # 1. Sınav türü kontrolü
        sinav_turu = data.get("sinav_turu", "").lower()
        logger.info(f"Tahmin isteği alındı: Sınav Türü={sinav_turu}")
        
        if sinav_turu not in GECERLI_SINAV_TURLERI:
            logger.error(f"Geçersiz sınav türü: {sinav_turu}")
            return jsonify({
                "error": "Geçersiz sınav türü",
                "gecerli_turler": GECERLI_SINAV_TURLERI
            }), 400

        # 2. Hedef sıralama kontrolü
        try:
            hedef_siralama = int(data.get("hedef_siralama", 0))
            if hedef_siralama <= 0:
                logger.error(f"Geçersiz hedef sıralama: {hedef_siralama}")
                return jsonify({"error": "Hedef sıralama pozitif bir sayı olmalıdır"}), 400
        except (ValueError, TypeError) as e:
            logger.error(f"Hedef sıralama dönüşüm hatası: {str(e)}")
            return jsonify({"error": "Hedef sıralama sayısal bir değer olmalıdır"}), 400

        # 3. Doğru/Yanlış verileri kontrolü
        dogru_yanlis_verileri = data.get("dogru_yanlis")
        if not isinstance(dogru_yanlis_verileri, dict) or not dogru_yanlis_verileri:
            logger.error("Doğru/yanlış verileri eksik veya hatalı format")
            return jsonify({"error": "Doğru/yanlış verileri eksik ya da hatalı formatta"}), 400

        # Yeni eklenen Adım: Gerekli derslerin kontrolü
        gerekli_dersler = set(Config.REQUIRED_LESSONS.get(sinav_turu, []))
        gelen_dersler = set(dogru_yanlis_verileri.keys())
        
        if not gerekli_dersler.issubset(gelen_dersler):
            eksik_dersler = list(gerekli_dersler - gelen_dersler)
            logger.error(f"Eksik dersler: {eksik_dersler}")
            return jsonify({
                "error": "Eksik ders bilgisi",
                "eksik_dersler": eksik_dersler
            }), 400

        # 4. Net Hesaplama
        try:
            netler = {}
            for ders, dy in dogru_yanlis_verileri.items():
                netler[ders] = net_hesapla(sinav_turu, dy)
            toplam_net = toplam_net_hesapla(sinav_turu, netler)
            logger.info(f"Net hesaplama tamamlandı: Toplam Net={toplam_net}")
        except Exception as e:
            logger.error(f"Net hesaplama hatası: {str(e)}")
            return jsonify({"error": f"Net hesaplama hatası: {str(e)}"}), 500

        # 5. Sıralama Tahmini
        try:
            tahmini_siralama = tahmin_yap(
                sinav_turu=sinav_turu,
                dogru_yanlis=dogru_yanlis_verileri,
                hedef_siralama=hedef_siralama
            )
            logger.info(f"Tahmin tamamlandı: Tahmini Sıralama={tahmini_siralama['siralama']['tahmini']}")
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return jsonify({"error": f"Tahmin hatası: {str(e)}"}), 500

        # 6. Sonuç Değerlendirmesi
        tahmin_sonucu = tahmini_siralama
        response = {
            "sinav_turu": sinav_turu,
            "net": tahmin_sonucu["net"],
            "siralama": tahmin_sonucu["siralama"]
        }
        
        logger.info(f"İşlem başarıyla tamamlandı: {sinav_turu}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {str(e)}", exc_info=True)
        return jsonify({"error": "Sunucu hatası oluştu"}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200
