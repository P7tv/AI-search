import streamlit as st
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import torch
import random
import shutil
from dotenv import load_dotenv
from github import Github
import base64
import pickle

# Load environment variables from .env file
load_dotenv()

# Get admin credentials from environment variables
if "ADMIN_USERNAME" in st.secrets and "ADMIN_PASSWORD" in st.secrets:
    ADMIN_USERNAME = st.secrets["ADMIN_USERNAME"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
else:
    st.error("Admin credentials not found in secrets. Please check your secrets.toml file.")
    st.stop()

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
@st.cache_data(ttl=3600)  # cache นาน 1 ชั่วโมง
def load_database_images(database_dir, _face_analyzer, use_cache=True):
    # ตรวจสอบไฟล์ cache
    cache_file = os.path.join(database_dir, "embedding.pkl")
    
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load cache: {str(e)}")
    
    database = []
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(list(Path(database_dir).glob(ext)))
    
    # แสดง progress bar
    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    
    total_images = len(all_images)
    for idx, img_path in enumerate(all_images):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = preprocess_image(img, max_size=320)
                faces = _face_analyzer.get(img)
                if faces:
                    database.append({
                        "path": str(img_path),
                        "embeddings": [face.normed_embedding for face in faces]
                    })
            
            # อัพเดท progress
            progress = (idx + 1) / total_images
            progress_bar.progress(progress)
            progress_text.text(f"Processing images: {int(progress * 100)}%")
            
        except Exception as e:
            st.sidebar.error(f"Error loading {img_path}: {str(e)}")
            continue
    
    progress_bar.empty()
    progress_text.empty()
    
    # บันทึก cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(database, f)
    except Exception as e:
        st.warning(f"Could not save cache: {str(e)}")
    
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
    filtered_matches = [m for m in matches if m.get('similarity', 0) * 100 >= 10]
    
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
                    
                    # เพิ่มปุ่ม Download
                    with open(match["path"], "rb") as file:
                        btn = st.download_button(
                            label="Download",
                            data=file,
                            file_name=os.path.basename(match["path"]),
                            mime="image/jpeg"
                        )

def create_new_album(album_name):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        os.makedirs(album_dir)
        st.success(f"Created new album: {album_name}")
    else:
        st.warning(f"Album '{album_name}' already exists")

# เพิ่มฟังก์ชันสำหรับจัดการ permanent embeddings
def get_permanent_embeddings_path(album_name):
    # เก็บ embeddings ไว้ใน permanent_embeddings folder
    permanent_dir = "permanent_embeddings"
    if not os.path.exists(permanent_dir):
        os.makedirs(permanent_dir)
    return os.path.join(permanent_dir, f"{album_name}_embeddings.pkl")

def save_permanent_embeddings(album_name, database):
    embedding_file = get_permanent_embeddings_path(album_name)
    try:
        with open(embedding_file, 'wb') as f:
            pickle.dump(database, f)
        st.sidebar.success(f"Saved permanent embeddings for {album_name}")
        return True
    except Exception as e:
        st.sidebar.error(f"Error saving permanent embeddings: {str(e)}")
        return False

def load_permanent_embeddings(album_name):
    embedding_file = get_permanent_embeddings_path(album_name)
    if os.path.exists(embedding_file):
        try:
            with open(embedding_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.sidebar.error(f"Error loading permanent embeddings: {str(e)}")
    return None

# ปรับปรุงฟังก์ชัน upload_to_album
def upload_to_album(album_name, uploaded_files, face_analyzer):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    uploaded_count = 0
    for uploaded_file in uploaded_files:
        file_path = os.path.join(album_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_count += 1
    
    st.success(f"Uploaded {uploaded_count} files to {album_name}")
    
    # สร้าง embeddings ใหม่และบันทึกเป็น permanent
    database = load_database_images(album_dir, face_analyzer, use_cache=False)
    save_permanent_embeddings(album_name, database)
    return database

def upload_folder_to_album(album_name, folder_path, face_analyzer):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    if not os.path.isdir(folder_path):
        st.error(f"Invalid folder path: {folder_path}")
        return
    
    uploaded_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(album_dir, file)
                shutil.copy(src_path, dest_path)
                uploaded_files.append(file)
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files to {album_name}")
    else:
        st.warning("No valid image files found in the folder")
    return reload_database(album_name, face_analyzer)

def view_album_images(album_name):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    images = [f for f in os.listdir(album_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        st.warning(f"No images found in album '{album_name}'")
        return
    
    st.write(f"Found {len(images)} images in album '{album_name}'")
    
    # แสดงรูปภาพในแถวละ 3 รูป
    cols = 3
    for i in range(0, len(images), cols):
        columns = st.columns(cols)
        for j, img_name in enumerate(images[i:i+cols]):
            with columns[j]:
                img_path = os.path.join(album_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img, caption=img_name)

def check_admin(username, password):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        return True
    else:
        st.session_state.is_admin = False
        return False

def reload_database(album_name, face_analyzer):
    database_dir = os.path.join("albums", album_name)
    return load_database_images(database_dir, face_analyzer)

def delete_album(album_name):
    album_dir = os.path.join("albums", album_name)
    if os.path.exists(album_dir):
        shutil.rmtree(album_dir)
        st.success(f"Deleted album: {album_name}")
    else:
        st.error(f"Album '{album_name}' does not exist")

def upload_to_github(album_name, file_name, file_data):
    try:
        # Initialize GitHub
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # Create path in GitHub
        path = f"AI-search/albums/{album_name}/{file_name}"
        
        # Encode file data to base64
        encoded_data = base64.b64encode(file_data).decode("utf-8")
        
        # Create or update file in GitHub
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, f"Update {file_name}", encoded_data, contents.sha)
        except:
            repo.create_file(path, f"Add {file_name}", encoded_data, branch="main")
        
        return True
    except Exception as e:
        st.error(f"Error uploading to GitHub: {str(e)}")
        return False

def upload_all_to_github(github_token, album_name):
    try:
        # Initialize GitHub
        g = Github(github_token)
        repo = g.get_repo(GITHUB_REPO)
        
        # Get album directory
        album_dir = os.path.join("albums", album_name)
        if not os.path.exists(album_dir):
            st.error(f"Album '{album_name}' does not exist")
            return False
        
        # Get all images in the album
        images = [f for f in os.listdir(album_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            st.warning(f"No images found in album '{album_name}'")
            return False
        
        # Upload each image to GitHub
        for img_name in images:
            img_path = os.path.join(album_dir, img_name)
            with open(img_path, "rb") as f:
                file_data = f.read()
            
            # Create path in GitHub
            path = f"AI-search/albums/{album_name}/{img_name}"
            
            # Encode file data to base64
            encoded_data = base64.b64encode(file_data).decode("utf-8")
            
            # Create or update file in GitHub
            try:
                contents = repo.get_contents(path)
                repo.update_file(path, f"Update {img_name}", encoded_data, contents.sha)
            except:
                repo.create_file(path, f"Add {img_name}", encoded_data, branch="main")
        
        return True
    except Exception as e:
        st.error(f"Error uploading to GitHub: {str(e)}")
        return False

# ปรับปรุงฟังก์ชัน lazy_load_database
def lazy_load_database(album_name, face_analyzer):
    # ลองโหลด permanent embeddings ก่อน
    database = load_permanent_embeddings(album_name)
    if database is not None:
        st.sidebar.success(f"Loaded permanent embeddings for {album_name}")
        return database
    
    # ถ้าไม่มี permanent embeddings ให้สร้างใหม่
    database_dir = os.path.join("albums", album_name)
    database = load_database_images(database_dir, face_analyzer)
    
    # บันทึกเป็น permanent embeddings
    save_permanent_embeddings(album_name, database)
    return database

# เพิ่มปุ่มสำหรับ Admin ในการจัดการ permanent embeddings
def add_admin_embedding_controls():
    if st.session_state.is_admin:
        st.sidebar.header("Embeddings Management")
        
        # ปุ่มสร้าง permanent embeddings ใหม่
        if st.sidebar.button("Regenerate Permanent Embeddings"):
            album_names = [name for name in os.listdir("albums") 
                         if os.path.isdir(os.path.join("albums", name))]
            
            progress_text = st.sidebar.empty()
            progress_bar = st.sidebar.progress(0)
            
            for idx, album_name in enumerate(album_names):
                progress_text.text(f"Processing {album_name}...")
                database_dir = os.path.join("albums", album_name)
                database = load_database_images(database_dir, face_analyzer, use_cache=False)
                save_permanent_embeddings(album_name, database)
                
                progress = (idx + 1) / len(album_names)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            progress_text.empty()
            st.sidebar.success("Regenerated all permanent embeddings")

def main():
    st.set_page_config(
        page_title="Face Recognition",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("Face Recognition App 👤")
    
    # Tutorial section
    st.header("Tutorial: การใช้งานแอปพลิเคชันค้นหารูปภาพด้วยใบหน้า")
    st.write("""
    ### 1. **เข้าสู่แอปพลิเคชัน**
    - เปิดเบราว์เซอร์และไปที่ลิงก์ของแอปพลิเคชัน
    - คุณจะเห็นหน้าแรกของแอปพลิเคชันพร้อมกับคำอธิบายสั้นๆ เกี่ยวกับการใช้งาน

    ### 2. **เลือกอัลบั้ม**
    - ในแถบด้านข้าง ให้เลือกอัลบั้มที่คุณต้องการค้นหารูปภาพ
    - หากไม่มีอัลบั้มที่ต้องการ คุณสามารถสร้างอัลบั้มใหม่ได้โดยเข้าสู่ระบบเป็น Admin

    ### 3. **อัปโหลดรูปภาพ**
    - คลิกที่ปุ่ม "Upload an image" เพื่ออัปโหลดรูปภาพที่ต้องการค้นหา
    - เลือกรูปภาพจากอุปกรณ์ของคุณ (รองรับไฟล์ .jpg, .jpeg, .png)

    ### 4. **รอการประมวลผล**
    - ระบบจะทำการประมวลผลรูปภาพที่คุณอัปโหลดและแสดงผลลัพธ์
    - คุณจะเห็นรูปภาพที่คล้ายกันพร้อมกับเปอร์เซ็นต์ความคล้ายคลึง

    ### 5. **ดูผลลัพธ์**
    - ผลลัพธ์จะแสดงเป็นรูปภาพที่คล้ายกันในแถวละ 3 รูป
    - คุณสามารถคลิกที่รูปภาพเพื่อดูรายละเอียดเพิ่มเติม

    ### 6. **ปรับแต่งการค้นหา**
    - คุณสามารถปรับค่า "Similarity Threshold" ในแถบด้านข้างเพื่อกำหนดระดับความคล้ายคลึงที่ต้องการ
    - ค่าเริ่มต้นคือ 25% คุณสามารถปรับค่าได้ตั้งแต่ 0% ถึง 100%

    ### 7. **ดูรูปภาพในอัลบั้ม**
    - คลิกที่ปุ่ม "View Album Images" ในแถบด้านข้างเพื่อดูรูปภาพทั้งหมดในอัลบั้มที่เลือก
    - คุณจะเห็นรูปภาพทั้งหมดในอัลบั้มที่แสดงในแถวละ 3 รูป

    ### 8. **ออกจากแอปพลิเคชัน**
    - เมื่อคุณใช้งานเสร็จสิ้น คุณสามารถปิดเบราว์เซอร์เพื่อออกจากแอปพลิเคชัน
    """)
    
    # Initialize components
    status = st.empty()
    progress = st.progress(0)
    face_analyzer = init_face_analyzer()
    
    # Initialize session state for login
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    
    # Login section
    st.sidebar.header("Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if check_admin(username, password):
            st.sidebar.success("Logged in as Admin")
        else:
            st.sidebar.error("Invalid username or password")
    
    # Logout button
    if st.session_state.is_admin:
        if st.sidebar.button("Logout"):
            st.session_state.is_admin = False
            st.sidebar.success("Logged out successfully")
    
    # Rest of the app logic
    if st.session_state.is_admin:
        # Admin functionality
        st.sidebar.header("Admin Panel")
        
        # GitHub token input
        github_token = st.sidebar.text_input("Enter GitHub Token", type="password")
        
        # Create new album section
        st.sidebar.header("Create New Album")
        new_album_name = st.sidebar.text_input("Enter album name")
        if st.sidebar.button("Create Album"):
            if new_album_name:
                create_new_album(new_album_name)
            else:
                st.sidebar.warning("Please enter an album name")
        
        # Upload to album section
        st.sidebar.header("Upload to Album")
        album_names = [name for name in os.listdir("albums") if os.path.isdir(os.path.join("albums", name))]
        selected_album = st.sidebar.selectbox("Select album", album_names, key="select_album_upload")
        # Multiple file upload
        uploaded_files = st.sidebar.file_uploader(
            "Choose images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        if uploaded_files and selected_album:
            database = upload_to_album(selected_album, uploaded_files, face_analyzer)
        
        # Folder upload
        folder_path = st.sidebar.text_input("Enter folder path to upload")
        if st.sidebar.button("Upload Folder") and selected_album and folder_path:
            database = upload_folder_to_album(selected_album, folder_path, face_analyzer)
        
        # Delete album section
        st.sidebar.header("Delete Album")
        album_to_delete = st.sidebar.selectbox("Select album to delete", album_names, key="select_album_delete")
        if st.sidebar.button("Delete Album"):
            delete_album(album_to_delete)
        
        # Upload to GitHub button
        if st.sidebar.button("Upload to GitHub") and github_token:
            if upload_all_to_github(github_token, selected_album):
                st.sidebar.success("Uploaded all images to GitHub")
            else:
                st.sidebar.error("Failed to upload images to GitHub")
    else:
        st.sidebar.warning("Please login as Admin to access admin features")
    
    # Load database from selected album แบบ lazy
    status.text("Loading database...")
    album_names = [name for name in os.listdir("albums") if os.path.isdir(os.path.join("albums", name))]
    selected_album = st.sidebar.selectbox("Select album", album_names, key="select_album_main")
    
    if selected_album:
        database = lazy_load_database(selected_album, face_analyzer)
        st.sidebar.write(f"Database images: {len(database)}")
    else:
        database = []
        st.sidebar.write("No album selected")
    
    # UI Controls
    with st.sidebar:
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25
        )
        
        # เพิ่มปุ่มดูรูปภาพใน albums
        if st.sidebar.button("View Album Images"):
            view_album_images(selected_album)

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

    # เพิ่มปุ่มสำหรับ Admin ในการจัดการ permanent embeddings
    add_admin_embedding_controls()

if __name__ == "__main__":
    main() 