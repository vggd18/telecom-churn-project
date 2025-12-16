import os
from datetime import datetime

def get_timestamped_path(filepath):
    """
    Adiciona data/hora ao nome do arquivo.
    Entrada: results/model.pkl
    Saída:   results/model_20251215_213000.pkl
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(directory, new_filename)