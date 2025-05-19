class Config:
    MODEL_PATH = "app/ml/models"
    DATA_PATH = "data/veri_setleri"
    DEBUG = False
    TESTING = False
    ALLOWED_EXAM_TYPES = ["tyt", "ayt_ea", "ayt_say", "ayt_soz"]
class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    DATA_PATH = "tests/test_data"