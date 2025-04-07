import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tempfile
import os
import sqlite3
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Initialize the app
st.set_page_config(
    page_title="Spitting Prevention System",
    page_icon="🚫",
    layout="wide"
)

# App title and description
st.title("🚫 Spitting Prevention System")
st.markdown("""
This application detects spitting behavior and displays the Aadhaar information of individuals caught spitting.
""")

# Sidebar
st.sidebar.title("Options")
mode = st.sidebar.selectbox("Choose Detection Mode", ["Image", "Video", "Camera"])
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.1, 0.05)
display_aadhaar = True
aadhaar_opacity = 0.7

# Database setup
def setup_database():
    """Create SQLite database for person-aadhaar mappings"""
    conn = sqlite3.connect('aadhaar_database.db')
    cursor = conn.cursor()
    
    # Create tables for person records and detection logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY,
        person_id TEXT UNIQUE,
        name TEXT,
        aadhaar_path TEXT,
        face_embedding BLOB
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_logs (
        id INTEGER PRIMARY KEY,
        person_id TEXT,
        timestamp TEXT,
        location TEXT,
        image_path TEXT,
        FOREIGN KEY (person_id) REFERENCES persons(person_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Create mock data for demonstration
def create_mock_data():
    """Create mock data for demonstration purposes"""
    conn = sqlite3.connect('aadhaar_database.db')
    cursor = conn.cursor()
    
    # Create aadhaar_cards directory if it doesn't exist
    os.makedirs("aadhaar_cards", exist_ok=True)
    
    # Sample data - in a real system, this would be securely managed
    sample_persons = [
        ("P001", "Raj Kumar", "aadhaar_cards/raj_kumar.jpg", None),
        ("P002", "Priya Singh", "aadhaar_cards/priya_singh.jpg", None),
        ("P003", "Amit Patel", "aadhaar_cards/amit_patel.jpg", None),
        ("P004", "Meera Shah", "aadhaar_cards/meera_shah.jpg", None),
        ("P005", "Vikram Malhotra", "aadhaar_cards/vikram_malhotra.jpg", None)
    ]
    
    # Create sample Aadhaar cards (blank images with text for demo)
    for person_id, name, aadhaar_path, _ in sample_persons:
        # Create a blank image with person details
        img = np.ones((400, 650, 3), dtype=np.uint8) * 255
        
        # Add Aadhaar card formatting
        cv2.rectangle(img, (10, 10), (640, 390), (0, 0, 200), 2)
        cv2.putText(img, "AADHAAR", (250, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, f"Name: {name}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, f"ID: XXXX-XXXX-{person_id}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "DOB: XX/XX/XXXX", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Address: XXXX XXXX XXXX", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "DEMO - NOT REAL AADHAAR", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # Save the mock Aadhaar card
        cv2.imwrite(aadhaar_path, img)
        
        # Insert or update the record
        cursor.execute('''
        INSERT OR REPLACE INTO persons (person_id, name, aadhaar_path)
        VALUES (?, ?, ?)
        ''', (person_id, name, aadhaar_path))
    
    conn.commit()
    conn.close()

# Get a list of available YOLO models
@st.cache_data
def get_available_models():
    # In a real app, you might scan a directory for models
    # For now, we'll hardcode a few options including your trained model
    return {
        "Spitting Detection model": "/Users/todankar/Desktop/spitting-detection-project/best.pt"
    }

# Random person identification (for demo purposes)
def identify_person(face_img=None):
    """
    Identify a person from their face image
    In a real system, this would use face recognition
    For demo, we'll randomly assign an identity
    """
    conn = sqlite3.connect('aadhaar_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT person_id, name, aadhaar_path FROM persons")
    persons = cursor.fetchall()
    conn.close()
    
    # In a real system, do face matching here
    # For demo, we'll randomly select a person
    import random
    selected_person = random.choice(persons)
    return selected_person

# Log detection to database
def log_detection(person_id, location="Unknown", image_path=None):
    """Log a spitting detection event"""
    conn = sqlite3.connect('aadhaar_database.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
    INSERT INTO detection_logs (person_id, timestamp, location, image_path)
    VALUES (?, ?, ?, ?)
    ''', (person_id, timestamp, location, image_path))
    
    conn.commit()
    conn.close()

# Process image function with Aadhaar display
def process_image_with_aadhaar(image, model, conf=0.4):
    # Convert PIL image to numpy array for OpenCV
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = np.array(Image.open(image))
    
    # Convert RGB to BGR (OpenCV uses BGR)
    if image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection
    results = model.predict(source=image_np, conf=conf, verbose=False)
    
    # Get the annotated image from YOLO
    annotated_image = results[0].plot()
    
    # Check if any spitting detection (assuming class 0 is spitting)
    spitting_detected = False
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if cls == 0:  # Assuming class 0 is spitting
                spitting_detected = True
                
                # If Aadhaar display is enabled
                if display_aadhaar:
                    # Get the bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get a face region (simplified for demo)
                    face_img = image_np[max(0, y1):min(y2, image_np.shape[0]), 
                                      max(0, x1):min(x2, image_np.shape[1])]
                    
                    if face_img.size > 0:
                        # Identify person
                        person_id, name, aadhaar_path = identify_person(face_img)
                        
                        # Get Aadhaar image
                        if os.path.exists(aadhaar_path):
                            aadhaar_img = cv2.imread(aadhaar_path)
                            
                            # Resize Aadhaar card
                            card_height = int(annotated_image.shape[0] * 0.3)
                            card_width = int(card_height * aadhaar_img.shape[1] / aadhaar_img.shape[0])
                            aadhaar_resized = cv2.resize(aadhaar_img, (card_width, card_height))
                            
                            # Create overlay area
                            overlay_y = annotated_image.shape[0] - card_height - 10
                            overlay_x = annotated_image.shape[1] - card_width - 10
                            
                            # Create semi-transparent overlay
                            overlay = annotated_image.copy()
                            cv2.rectangle(overlay, (overlay_x-10, overlay_y-40), 
                                        (overlay_x+card_width+10, overlay_y+card_height+10), 
                                        (0, 0, 0), -1)
                            
                            # Add transparency
                            annotated_image = cv2.addWeighted(overlay, aadhaar_opacity, annotated_image, 1-aadhaar_opacity, 0)
                            
                            # Add Aadhaar card to frame
                            annotated_image[overlay_y:overlay_y+card_height, 
                                          overlay_x:overlay_x+card_width] = aadhaar_resized
                            
                            # Add person name
                            cv2.putText(annotated_image, f"Name: {name}", 
                                        (overlay_x, overlay_y-15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Log the detection (optional)
                            log_detection(person_id, location="Streamlit App")
    
    # Convert back to RGB for Streamlit display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image_rgb, spitting_detected

# Process video function with Aadhaar display
def process_video_with_aadhaar(video_path, model, conf=0.4):
    cap = cv2.VideoCapture(video_path)
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.mp4")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    spitting_detected = False
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        frame_spitting = False
        
        # Check if any spitting detection
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                if cls == 0:  # Assuming class 0 is spitting
                    spitting_detected = True
                    frame_spitting = True
                    
                    # If Aadhaar display is enabled
                    if display_aadhaar:
                        # Get the bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get a face region (simplified for demo)
                        face_img = frame[max(0, y1):min(y2, frame.shape[0]), 
                                       max(0, x1):min(x2, frame.shape[1])]
                        
                        if face_img.size > 0:
                            # Identify person
                            person_id, name, aadhaar_path = identify_person(face_img)
                            
                            # Get Aadhaar image
                            if os.path.exists(aadhaar_path):
                                aadhaar_img = cv2.imread(aadhaar_path)
                                
                                # Resize Aadhaar card
                                card_height = int(frame.shape[0] * 0.3)
                                card_width = int(card_height * aadhaar_img.shape[1] / aadhaar_img.shape[0])
                                aadhaar_resized = cv2.resize(aadhaar_img, (card_width, card_height))
                                
                                # Create overlay area
                                overlay_y = frame.shape[0] - card_height - 10
                                overlay_x = frame.shape[1] - card_width - 10
                                
                                # Create semi-transparent overlay
                                overlay = annotated_frame.copy()
                                cv2.rectangle(overlay, (overlay_x-10, overlay_y-40), 
                                            (overlay_x+card_width+10, overlay_y+card_height+10), 
                                            (0, 0, 0), -1)
                                
                                # Add transparency
                                annotated_frame = cv2.addWeighted(overlay, aadhaar_opacity, annotated_frame, 1-aadhaar_opacity, 0)
                                
                                # Add Aadhaar card to frame
                                annotated_frame[overlay_y:overlay_y+card_height, 
                                              overlay_x:overlay_x+card_width] = aadhaar_resized
                                
                                # Add person name
                                cv2.putText(annotated_frame, f"Name: {name}", 
                                            (overlay_x, overlay_y-15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Log the detection (every 10th frame to avoid database bloat)
                                if frame_count % 10 == 0:
                                    log_detection(person_id, location="Video Processing")
        
        # Write frame to output video
        out.write(annotated_frame)
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path, spitting_detected

# View detection logs
def view_detection_logs():
    conn = sqlite3.connect('aadhaar_database.db')
    cursor = conn.cursor()
    
    # Join tables to get person information with logs
    cursor.execute('''
    SELECT d.id, d.timestamp, p.name, p.person_id, d.location
    FROM detection_logs d
    JOIN persons p ON d.person_id = p.person_id
    ORDER BY d.timestamp DESC
    LIMIT 100
    ''')
    
    logs = cursor.fetchall()
    conn.close()
    
    return logs

# Setup database and mock data
setup_database()
create_mock_data()

# Get available models
models = get_available_models()
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model_path = models[selected_model_name]

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the selected model
model = load_model(selected_model_path)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Detection", "Detection Logs", "About"])

with tab1:
    # Detection mode based on selection
    if mode == "Image":
        st.header("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Perform detection with Aadhaar display
            if st.button("Detect Spitting"):
                with st.spinner("Processing image..."):
                    annotated_image, spitting_detected = process_image_with_aadhaar(
                        uploaded_image, model, conf=confidence
                    )
                    
                    st.image(annotated_image, caption="Detection Result", use_column_width=True)
                    
                    if spitting_detected:
                        st.error("⚠️ Spitting Detected!")
                    else:
                        st.success("✅ No Spitting Detected")

    elif mode == "Video":
        st.header("Upload a Video")
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_video.read())
                temp_video_path = temp_file.name
            
            # Display the original video
            st.video(uploaded_video)
            
            # Process the video
            if st.button("Detect Spitting in Video"):
                with st.spinner("Processing video... This may take a while."):
                    output_video_path, spitting_detected = process_video_with_aadhaar(
                        temp_video_path, model, conf=confidence
                    )
                    
                    # Display the processed video
                    st.video(output_video_path)
                    
                    if spitting_detected:
                        st.error("⚠️ Spitting Detected in Video!")
                    else:
                        st.success("✅ No Spitting Detected in Video")
                    
                    # Clean up
                    try:
                        os.unlink(temp_video_path)
                    except:
                        pass

    elif mode == "Camera":
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        import av

class SpittingDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(source=img, conf=confidence, verbose=False)
        annotated = results[0].plot()

        # Optional: Aadhaar overlay here if spitting detected
        return annotated

with tab1:
    if mode == "Camera":
        st.header("Live Camera Detection")
        st.warning("Note: You must allow camera access in your browser.")

        webrtc_streamer(
            key="spit-detect",
            video_transformer_factory=SpittingDetector,
            media_stream_constraints={"video": True, "audio": False},
        )

        
        cam_test_image = st.file_uploader("Upload a camera test image (optional)", type=["jpg", "jpeg", "png"])

        if cam_test_image is not None:
            image = Image.open(cam_test_image)
            st.image(image, caption="Test Image", use_column_width=True)
            
            if st.button("Detect Spitting (Camera Simulation)"):
                with st.spinner("Processing image..."):
                    annotated_image, spitting_detected = process_image_with_aadhaar(
                        cam_test_image, model, conf=confidence
                    )

                    st.image(annotated_image, caption="Detection Result", use_column_width=True)

                    if spitting_detected:
                        st.error("⚠️ Spitting Detected!")
                    else:
                        st.success("✅ No Spitting Detected")


with tab2:
    st.header("Detection Logs")
    
    if st.button("Refresh Logs"):
        logs = view_detection_logs()
        
        if logs:
            # Create a DataFrame for better display
            import pandas as pd
            log_df = pd.DataFrame(logs, columns=["ID", "Timestamp", "Name", "Person ID", "Location"])
            st.dataframe(log_df)
        else:
            st.info("No detection logs found")

with tab3:
    st.header("About the System")
    st.markdown("""
    ### Spitting Prevention System
    
    This application uses computer vision to detect spitting behavior and identify individuals through their Aadhaar cards.
    
    **Features:**
    - Detect spitting in images and videos
    - Display Aadhaar information of offenders
    - Log incidents with timestamps
    - Support for multiple detection models
    
    **Notes:**
    - This is a demonstration application
    - The Aadhaar cards shown are mock representations for demo purposes
    - In a real-world application, proper authorization and privacy measures would be required
    
    **Privacy Considerations:**
    In a production environment, this system would need to comply with relevant privacy laws and regulations,
    including obtaining proper authorization for displaying identification documents.
    """)

    st.subheader("How it Works")
    st.markdown("""
    1. The system uses a trained YOLO model to detect spitting behavior
    2. When detected, it identifies the individual (simulated in this demo)
    3. The system displays their Aadhaar card information
    4. All incidents are logged in a database for future reference
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Team [Matrix]")