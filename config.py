class Config:
    MODEL_PATH = "data/models/v3"
    DATA_PATH = "data/veri_setleri/processed_v3"
    DEBUG = False
    TESTING = False
    ALLOWED_EXAM_TYPES = ["tyt", "ayt_ea", "ayt_sayisal", "ayt_sozel", "ayt_dil"]
    
    # Her sınav türü için zorunlu olan dersler
    REQUIRED_LESSONS = {
        "tyt": ["tyt_turkce", "tyt_matematik", "tyt_fen", "tyt_sosyal"],
        "ayt_sayisal": ["ayt_matematik", "ayt_fizik", "ayt_kimya", "ayt_biyoloji"],
        "ayt_ea": ["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"],
        "ayt_sozel": ["ayt_edebiyat", "ayt_cografya1"],
        "ayt_dil": ["ayt_dil"]
    }

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    DATA_PATH = "tests/test_data"