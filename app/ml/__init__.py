from app.ml.train import train_models

def init_models():
    """Modelleri eğit veya yükle"""
    try:
        train_models()
    except Exception as e:
        print(f"Model eğitimi hatası: {str(e)}")
        raise