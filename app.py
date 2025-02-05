import os
import pickle
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize face analyzer with caching
@st.cache_resource
def init_face_analyzer():
    # ตรวจสอบการใช้งาน MPS
    if torch.backends.mps.is_available():
        st.sidebar.success("Using M2 GPU (Metal)")
        # ใช้ MPS backend
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=['MPSExecutionProvider'],
            allowed_modules=['detection', 'recognition']  # เพิ่มโมดูลที่ใช้ GPU
        )
        analyzer.prepare(ctx_id=0)  # ใช้ GPU
    else:
        st.sidebar.info("Using CPU")
        # Fallback to CPU
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )
        analyzer.prepare(ctx_id=-1)
    
    return analyzer

# Cache database loading
@st.cache_data
def load_database_images(database_dir, _face_analyzer, max_samples=200):
    """โหลดรูปและ random sampling ให้เหลือไม่เกิน max_samples"""
    database = []
    
    # ใช้ progress bar
    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    
    # รวบรวมรูปทั้งหมด
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(list(Path(database_dir).glob(ext)))
    
    # สุ่มเลือกรูป
    if len(all_images) > max_samples:
        st.sidebar.warning(f"Sampling {max_samples} images from {len(all_images)} total images")
        selected_images = random.sample(all_images, max_samples)
    else:
        selected_images = all_images
    
    # ใช้ batch processing
    batch_size = 10
    total_batches = len(selected_images) // batch_size + 1
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(selected_images))
        batch_images = selected_images[start_idx:end_idx]
        
        # อัพเดท progress
        progress = (batch_idx + 1) / total_batches
        progress_text.text(f"Loading images: {int(progress * 100)}%")
        progress_bar.progress(progress)
        
        # Process batch
        for img_path in batch_images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Resize ให้เล็กลงเพื่อเร็วขึ้น
                    img = preprocess_image(img, max_size=320)  # ลดขนาดลง
                    faces = _face_analyzer.get(img)
                    if faces:
                        database.append({
                            "path": str(img_path),
                            "embeddings": [face.normed_embedding for face in faces]
                        })
            except Exception as e:
                st.sidebar.error(f"Error loading {img_path}: {str(e)}")
                continue
    
    # Clear progress
    progress_text.empty()
    progress_bar.empty()
    
    return database

# Optimize image size
def preprocess_image(image, max_size=640):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size)
    return image

# Vectorized matching
@st.cache_data
def find_matches_vectorized(query_embedding, database, threshold):
    if not database or query_embedding is None:
        return []
    
    # Combine all embeddings into a single matrix
    all_embeddings = []
    all_paths = []
    for db in database:
        if db.get("embeddings"):  # ตรวจสอบว่ามี embeddings
            all_embeddings.extend(db["embeddings"])
            all_paths.extend([db["path"]] * len(db["embeddings"]))
    
    if not all_embeddings:
        return []
        
    try:
        # Calculate similarities in one go
        all_embeddings = np.vstack(all_embeddings)
        similarities = np.dot(all_embeddings, query_embedding)
        
        # Find matches above threshold
        mask = similarities > threshold
        matches = [{
            "path": path,
            "similarity": float(sim)
        } for path, sim in zip(np.array(all_paths)[mask], similarities[mask])]
        
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        st.error(f"Error in matching: {str(e)}")
        return []

# Efficient result display
def display_results(matches, cols=3):
    if not matches:
        return
    
    # กรองเฉพาะ matches ที่มีความคล้ายเกิน 25%
    filtered_matches = [m for m in matches if m.get('similarity', 0) * 100 >= 25]
    
    if not filtered_matches:
        st.write("No matches above 25% similarity")
        return
    
    # แสดงจำนวนรูปที่เจอ
    st.write(f"Found {len(filtered_matches)} matches above 25% similarity")
    
    # แบ่งการแสดงผลเป็นแถว แถวละ 3 รูป
    for i in range(0, len(filtered_matches), cols):
        columns = st.columns(cols)
        for j, match in enumerate(filtered_matches[i:i+cols]):
            with columns[j]:
                similarity_percent = f"{match['similarity']*100:.1f}%"
                match_img = cv2.imread(match["path"])
                if match_img is not None:
                    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                    st.image(match_img, caption=f"Similarity: {similarity_percent}")

def save_matched_model(database, model_path="matched_model.pkl"):
    """Save matched model to file"""
    with open(model_path, "wb") as f:
        pickle.dump(database, f)
    st.sidebar.success(f"Saved matched model to {model_path}")

def load_matched_model(model_path="matched_model.pkl"):
    """Load matched model from file"""
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

def main():
    ctx = get_script_run_ctx()
    if ctx is None:
        # Handle bare mode
        pass
    
    st.title("Zero-shot Face Recognition 👤")
    
    # Initialize components
    status = st.empty()
    progress = st.progress(0)
    face_analyzer = init_face_analyzer()
    
    # Load database
    status.text("Loading database...")
    database_dir = "database"
    Path(database_dir).mkdir(exist_ok=True)
    matched_model = load_matched_model()
    if matched_model:
        database = matched_model
        st.sidebar.info("Loaded pre-matched model")
    else:
        database = load_database_images(database_dir, face_analyzer)
        save_matched_model(database)  # Save after first match
    
    # UI Controls
    with st.sidebar:
        st.write(f"Database images: {len(database)}")
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25
        )

    status.text("Ready for image upload")
    progress.progress(100)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            status.text("Processing image...")
            progress.progress(30)
            
            # Read and preprocess image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = preprocess_image(image)
            
            # Detect faces
            status.text("Detecting faces...")
            faces = face_analyzer.get(image)
            progress.progress(60)
            
            if len(faces) > 0:
                status.text(f"Found {len(faces)} faces")
                
                # Create columns for results
                result_cols = st.columns(min(len(faces), 3))
                
                for i, (face, col) in enumerate(zip(faces, result_cols)):
                    with col:
                        # Show face info
                        gender = "Male" if face.sex == 1 else "Female"
                        age = int(face.age) if face.age is not None else 0
                        st.write(f"Face #{i+1}: {gender}, {age} years")
                        
                        # Find and display matches
                        matches = find_matches_vectorized(
                            face.normed_embedding, 
                            database, 
                            similarity_threshold
                        )
                        display_results(matches)
                
                status.text("Processing complete!")
                progress.progress(100)
                
            else:
                status.text("No faces detected")
                progress.progress(100)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            progress.progress(100)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Recognition",
        page_icon="👤",
        layout="wide"
    )
    main() 