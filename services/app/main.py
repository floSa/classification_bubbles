# Dashboard Streamlit - Dual Mode (Simplifié)
"""
Mode Training : 2 barres de progression (génération + entraînement)
Mode Realtime : Signal + FFT + Spectrogramme mis à jour toutes les secondes
"""

import os
import json
import sys
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_mongo_client, 
    get_minio_client,
    fetch_spectrogram_image,
    fetch_latest_bubble_complete
)

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Bubble Project",
    page_icon="🫧",
    layout="wide"
)

MODELS_DIR = "/app/models"
MODEL_FILE = "bubble_classifier.pth"
TRAINING_PROGRESS_FILE = os.path.join(MODELS_DIR, "training_progress.json")
ACQUISITION_PROGRESS_FILE = os.path.join(MODELS_DIR, "acquisition_progress.json")

# API Inference
INFERENCE_API_URL = "http://inference:8000"

# Intervalles de rafraîchissement
REFRESH_TRAINING_MS = 2000  # 2s pour le mode training
REFRESH_REALTIME_MS = 1000  # 1s pour le mode temps réel


# =============================================================================
# HELPERS
# =============================================================================

def load_json_file(filepath):
    """Charge un fichier JSON de manière sécurisée."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def is_training_finished() -> bool:
    """Vérifie si l'entraînement est terminé (100%)."""
    if not os.path.exists(TRAINING_PROGRESS_FILE):
        return False
        
    try:
        train_prog = load_json_file(TRAINING_PROGRESS_FILE)
        return train_prog.get("status") == "completed"
    except Exception:
        return False


def model_exists() -> bool:
    """Vérifie si le fichier modèle existe."""
    return os.path.exists(os.path.join(MODELS_DIR, MODEL_FILE))


def get_prediction_from_inference(bubble_id: str) -> dict:
    """
    Appelle le service inference pour obtenir la prédiction d'une bulle.
    Timeout très court pour ne pas bloquer le rafraîchissement.
    
    Args:
        bubble_id: ID MongoDB de la bulle
        
    Returns:
        dict avec predicted_label, confidence ou None si erreur/timeout
    """
    try:
        response = requests.get(
            f"{INFERENCE_API_URL}/predict/{bubble_id}",
            timeout=0.5  # Timeout court pour ne pas bloquer le refresh
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # Modèle non chargé
            return None
    except requests.exceptions.ConnectionError:
        # Service non disponible (normal au démarrage)
        pass
    except requests.exceptions.Timeout:
        # Timeout - utiliser le fallback MongoDB
        pass
    except Exception:
        pass
    return None


def determine_refresh_interval() -> int:
    """
    Détermine l'intervalle de rafraîchissement selon le mode actuel.
    
    Returns:
        Intervalle en millisecondes
    """
    # Si le modèle existe et training terminé → mode temps réel → 1s
    if model_exists() or is_training_finished():
        return REFRESH_REALTIME_MS
    return REFRESH_TRAINING_MS


# =============================================================================
# MODE TRAINING : Barres de progression uniquement
# =============================================================================

def render_training_mode():
    """Affiche UNIQUEMENT les deux barres de progression basées sur les JSONs."""
    
    st.title("🧠 Initialisation du Système")
    
    # === Barre 1 : Génération de Données (Phase 1) ===
    st.subheader("📦 Phase 1 : Génération de Données")
    
    acq_prog = load_json_file(ACQUISITION_PROGRESS_FILE)
    
    if acq_prog:
        total = acq_prog.get("total", 1)
        current = acq_prog.get("current", 0)
        percent = acq_prog.get("percentage", 0.0)
        status = acq_prog.get("status", "unknown")
        
        # Clamp value between 0.0 and 1.0
        progress_val = min(max(current / total, 0.0), 1.0) if total > 0 else 0.0
        
        st.progress(progress_val)
        st.caption(f"Progression : {percent}% ({current}/{total} chunks)")
        
        if status == "completed":
            st.success("✅ Génération terminée")
    else:
        st.info("⏳ En attente du service d'acquisition...")

    st.markdown("---")
    
    # === Barre 2 : Entraînement du Modèle (Phase 2) ===
    st.subheader("🔄 Phase 2 : Entraînement du Modèle")
    
    train_prog = load_json_file(TRAINING_PROGRESS_FILE)
    
    if train_prog:
        current_epoch = train_prog.get("current_epoch", 0)
        total_epochs = train_prog.get("total_epochs", 10)
        status = train_prog.get("status", "unknown")
        
        progress_val = min(current_epoch / total_epochs, 1.0) if total_epochs > 0 else 0.0
        
        st.progress(progress_val)
        st.caption(f"Epoch {current_epoch} / {total_epochs} ({progress_val * 100:.0f}%)")
        
        if status == "completed":
            st.success("✅ Entraînement terminé ! Redirection...")
        elif status == "training":
            st.info(f"🔄 Entraînement en cours... (Loss: {train_prog.get('loss', 0):.4f})")
        elif status == "waiting_data":
            msg = train_prog.get("message", "Attente de données...")
            st.warning(f"⏳ {msg}")
        elif status == "starting":
            st.info("🚀 Initialisation de l'entraînement...")
        else:
            st.info(f"Statut : {status}")
    else:
        # Si la génération n'est pas finie, le training n'a pas commencé
        if acq_prog.get("status") != "completed":
            st.caption("🔒 En attente de la fin de la génération...")
        else:
            st.info("⏳ Démarrage de l'entraînement...")


# =============================================================================
# MODE REALTIME : Signal + FFT + Spectrogramme
# =============================================================================

def render_realtime_mode():
    """Affiche le spectrogramme, signal et FFT du dernier échantillon."""
    
    st.title("🏭 Monitoring Temps Réel")
    
    # Init connexions (Lazy loading)
    mongo_client = get_mongo_client()
    minio_client = get_minio_client()
    
    # Récupérer la dernière bulle depuis MongoDB
    bubble = fetch_latest_bubble_complete(mongo_client)
    
    if not bubble:
        st.warning("⏳ En attente des premières données temps réel...")
        return
    
    # === Affichage des 3 graphiques ===
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔊 Signal")
        raw_audio = bubble.get("raw_audio", [])
        if raw_audio:
            df_signal = pd.DataFrame({"amplitude": raw_audio})
            st.line_chart(df_signal, height=300)
        else:
            st.warning("Pas de signal")
    
    with col2:
        st.subheader("📊 FFT")
        fft_data = bubble.get("fft_data", {})
        if fft_data and fft_data.get("magnitudes"):
            df_fft = pd.DataFrame({
                "Hz": fft_data.get("frequencies", []),
                "Mag": fft_data.get("magnitudes", [])
            })
            if not df_fft.empty:
                st.line_chart(df_fft.set_index("Hz"), height=300)
            else:
                st.warning("FFT vide")
        else:
            st.warning("FFT non disponible")
    
    with col3:
        st.subheader("🖼️ Spectrogramme")
        s3_key = bubble.get("s3_key")
        if s3_key:
            img = fetch_spectrogram_image(minio_client, s3_key)
            if img:
                st.image(img, width=300)
            else:
                st.warning("Image non disponible")
        else:
            st.warning("Aucun spectrogramme")
    
    # --- Prédiction via API Inference ---
    bubble_id = bubble.get("_id")
    label = bubble.get("label", "?")
    
    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.metric("Label Réel (Taux de bouchage)", f"{label}%")
    
    with col_p2:
        # Essayer d'obtenir la prédiction depuis l'API inference
        if bubble_id:
            prediction = get_prediction_from_inference(bubble_id)
            if prediction:
                st.metric(
                    "Prédiction", 
                    f"{prediction.get('predicted_label', '?')}", 
                    delta=f"{prediction.get('confidence', 0) * 100:.0f}% confiance"
                )
            else:
                # Fallback: utiliser la prédiction stockée dans MongoDB (si existe)
                mongo_pred = bubble.get("prediction")
                if mongo_pred:
                    st.metric(
                        "Prédiction", 
                        f"{mongo_pred.get('label')}%",
                        delta=f"{mongo_pred.get('confidence', 0) * 100:.0f}% confiance"
                    )
                else:
                    st.metric("Prédiction", "⏳ En attente")
        else:
            st.metric("Prédiction", "⏳ En attente")
    
    # Timestamp et ID
    ts = bubble.get("timestamp")
    bubble_id = bubble.get("_id", "?")
    
    st.markdown("---")
    if ts:
        now_str = datetime.now().strftime('%H:%M:%S')
        st.info(f"🔖 ID: `{bubble_id}` | 🕒 Timestamp Bulle: {ts.strftime('%H:%M:%S')} | 🔄 Refresh: {now_str}")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    try:
        # Déterminer l'intervalle de rafraîchissement dynamiquement
        refresh_ms = determine_refresh_interval()
        st_autorefresh(interval=refresh_ms, limit=None, key="auto_refresh")
        
        # 1. Charger l'état de l'acquisition
        acq_prog = load_json_file(ACQUISITION_PROGRESS_FILE)
        acq_status = acq_prog.get("status", "unknown")
        
        # 2. Si l'acquisition est en cours (ou pas commencée), on force le mode Training View
        if acq_status != "completed":
            render_training_mode()
        
        # 3. Si acquisition finie, on regarde si le training est fini OU si modèle existe
        elif is_training_finished() or model_exists():
            render_realtime_mode()
            
        # 4. Sinon (acquisition finie mais training pas fini), on reste en Training View
        else:
            render_training_mode()
            
    except Exception as e:
        st.error(f"❌ Erreur critique : {e}")
        import traceback
        st.code(traceback.format_exc())
