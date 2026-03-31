import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import open_clip
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="DeepSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom comprehensive CSS for a highly professional, modern slate theme
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
    }

    /* Suggestion Chips */
    .suggestion-btn {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        display: inline-block;
        font-size: 0.9em;
        font-weight: 600;
        color: #cbd5e1;
        transition: all 0.2s ease;
        text-align: center;
        cursor: pointer;
    }
    .suggestion-btn:hover {
        background-color: #2563eb;
        border-color: #3b82f6;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    /* Professional Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    label[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    /* Result Cards */
    .result-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
    }

    /* Thin Professional Divider */
    hr {
        border: 0;
        height: 1px;
        background-color: #1e293b;
        margin: 30px 0;
    }
    
    /* Input and Primary Button Styling */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add Logo to Top Corner
if os.path.exists("logo.png"):
    st.logo("logo.png")

# -------------------------------
# Load CLIP (Stable version)
# -------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer, device

model, preprocess, tokenizer, device = load_model()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    st.title("⚙️ Settings")
    st.markdown("Customize how video search operates.")
    frame_interval = st.slider("🎞️ Frame Interval (sec)", 1, 5, 2, help="Lower value extracts more frames but takes longer to process.")
    top_k = st.slider("📊 Max Results", 1, 20, 5, help="Number of matching clips to return.")
    st.divider()
    st.markdown("### About")
    st.info("Upload any video and use natural language to instantly find matching scenes using AI vision.")

# -------------------------------
# Extract frames
# -------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    frames = []
    timestamps = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % (fps * frame_interval) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(count / fps)

        count += 1

    cap.release()
    return frames, timestamps, fps, video_length

# -------------------------------
# Image embeddings
# -------------------------------
def get_image_embeddings(frames):
    embeddings = []

    for frame in frames:
        image = Image.fromarray(frame)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy()[0])

    return embeddings

# -------------------------------
# Text embedding
# -------------------------------
def get_text_embedding(text):
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        emb = model.encode_text(text_tokens)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# -------------------------------
# Matching
# -------------------------------
def find_matches(text_emb, image_embs, timestamps):
    scores = []

    for i, img_emb in enumerate(image_embs):
        similarity = np.dot(text_emb, img_emb)
        scores.append((float(similarity), timestamps[i], i))

    scores.sort(reverse=True)
    return scores[:top_k]

# -------------------------------
# Extract clip
# -------------------------------
def extract_clip(video_path, timestamp, fps, duration=3):
    cap = cv2.VideoCapture(video_path)

    start_frame = int((timestamp - duration/2) * fps)
    start_frame = max(start_frame, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(int(duration * fps)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    clip_path = f"clip_{int(timestamp)}.mp4"
    height, width, _ = frames[0].shape

    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for f in frames:
        out.write(f)

    out.release()
    return clip_path

# -------------------------------
# UI Core Application
# -------------------------------
if os.path.exists("logo.png"):
    col1, col2 = st.columns([1, 15])
    with col1:
        st.image("logo.png", width=65)
    with col2:
        st.markdown('<div class="gradient-text" style="line-height:65px; margin-top:-10px;">DeepSearch AI</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="gradient-text">🔍 DeepSearch AI</div>', unsafe_allow_html=True)

st.markdown("### 🎥 Natural Language Video Search")
st.write("Upload a video or use your live camera to seamlessly find any object, action, or scene within seconds using AI semantic search.")

# Initialize session state for live feed
if "live_frames" not in st.session_state:
    st.session_state["live_frames"] = []
if "live_timestamps" not in st.session_state:
    st.session_state["live_timestamps"] = []
if "live_embeddings" not in st.session_state:
    st.session_state["live_embeddings"] = []
import time
from datetime import datetime

tab1, tab2 = st.tabs(["📁 Video Upload", "📹 Live Camera Feed"])

with tab1:
    uploaded_file = st.file_uploader("📂 Upload your video file (MP4 format recommended)", type=["mp4", "avi", "mkv"])

    if uploaded_file:
        # Save the file
        video_path = "video_upload.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # --- Video Analysis Process ---
        st.divider()
        st.markdown("#### ⚙️ Video Processing")
        
        with st.status("Analyzing Video Content...", expanded=True) as status:
            st.write("🎞️ Extracting frames from video...")
            frames, timestamps, fps, v_length = extract_frames(video_path)
            
            st.write("🧠 Generative AI Model generating embeddings...")
            image_embs = get_image_embeddings(frames)
            
            status.update(label="✅ Video Analyzed Successfully!", state="complete", expanded=False)

        # --- Metrics Section ---
        cols = st.columns(4)
        with cols[0]:
            st.metric(label="Duration", value=f"{v_length:.1f}s")
        with cols[1]:
            st.metric(label="Total Frames", value=f"{len(frames)}")
        with cols[2]:
            st.metric(label="FPS", value=f"{fps}")
        with cols[3]:
            st.metric(label="Sampling Interval", value=f"{frame_interval}s")

        # Expandable preview of all frames
        with st.expander("🖼️ View Extracted Frame Gallery", expanded=False):
            st.markdown("Browse through all images that the AI analyzed:")
            gallery_cols = st.columns(6)
            for idx, img in enumerate(frames):
                with gallery_cols[idx % 6]:
                    st.image(img, caption=f"{timestamps[idx]:.1f}s", use_container_width=True)

        st.divider()

        # --- Search Interface ---
        st.markdown("### 🔎 Search The Scene")
        
        # Pre-defined suggestion chips using columns
        st.write("**💡 Try these queries or enter your own:**")
        suggestions_tab1 = ["🚶 Person walking", "🚗 A red car", "🐶 Animal present", "🌲 Trees outdoors", "🏢 Buildings", "💬 Text on screen"]
        
        # 6 columns for chips
        chip_cols = st.columns(len(suggestions_tab1))
        for i, sug in enumerate(suggestions_tab1):
            if chip_cols[i].button(sug, key=f"sug_tab1_{i}", use_container_width=True):
                st.session_state["search_query_tab1"] = sug.split(" ", 1)[-1]  # Remove emoji

        # Search Box
        col_input, col_submit = st.columns([4, 1])
        with col_input:
            query = st.text_input("Enter natural language query", key="query_tab1", value=st.session_state.get("search_query_tab1", ""), placeholder="e.g. Someone walking on the grass...", label_visibility="collapsed")
        with col_submit:
            search_pressed = st.button("🚀 Search", key="search_btn_tab1", type="primary", use_container_width=True)

        # Automatically search if query has changed via button OR search is pressed
        if search_pressed or query:
            st.session_state["search_query_tab1"] = query  # sync state
            with st.spinner(f"🔍 Searching for: **{query}**"):
                text_emb = get_text_embedding(query)
                results = find_matches(text_emb, image_embs, timestamps)

            if not results:
                st.warning("No matches found. Try a different query.")
            else:
                st.success(f"Found {len(results)} matching scenes!")
                st.markdown(f"## 🎯 Top Matches for '{query}'")

                for rank, (score, time_stamp, idx) in enumerate(results, 1):
                    # Create a visually pleasing container for each result
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'<h4 style="color:#e2e8f0;">🏆 Rank #{rank} <span style="font-size:0.8em; color:#818cf8;">(Match Score: {score:.2f})</span></h4>', unsafe_allow_html=True)
                    st.caption(f"📍 Location in video: **{time_stamp:.2f} seconds**")
                    
                    data_col_1, data_col_2 = st.columns(2)
                    
                    # Left side: Frame
                    with data_col_1:
                        st.markdown("**🖼️ Extracted Keyframe**")
                        st.image(frames[idx], use_container_width=True)

                    # Right side: Clip
                    with data_col_2:
                        st.markdown("**🎬 Contextual Clip**")
                        clip_path = extract_clip(video_path, time_stamp, fps)
                        if clip_path:
                            st.video(clip_path)
                        else:
                            st.info("Failed to extract context clip.")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Call to action when no video is uploaded
        st.info("👆 Please upload an MP4 video to get started.")

with tab2:
    st.markdown("### 🔴 Live Camera Capture")
    st.write("Record a short video clip using your local webcam and search through it instantly.")

    recording_duration = st.slider("⏱️ Recording Duration (seconds)", min_value=1, max_value=15, value=5)
    
    if st.button("🔴 Start Live Recording", type="primary", use_container_width=True):
        st.info(f"Starting your webcam to record for {recording_duration} seconds... Please wait.")
        
        # Access Webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open your webcam. Please check your permissions or if another app is using it.")
        else:
            # We want to show a live preview while recording
            preview_placeholder = st.empty()
            
            # Setup VideoWriter
            fps_cam = 20.0 # Common webcam FPS assumption, extract_frames will parse the actual later
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            live_path = "live_feed_recording.mp4"
            out = cv2.VideoWriter(live_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_cam, (width, height))
            
            start_time = time.time()
            with st.spinner(f"🔴 RECORDING LIVE FOR {recording_duration} SECONDS..."):
                while (time.time() - start_time) < recording_duration:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab a frame from the webcam.")
                        break
                    
                    # Write to file
                    out.write(frame)
                    
                    # Also update UI live preview
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
            # Release everything
            cap.release()
            out.release()
            preview_placeholder.empty()
            
            st.success(f"Successfully recorded a {recording_duration}-second clip!")
            st.session_state["live_video_path"] = live_path

    # If we have a recorded live video in session, process it exactly like uploaded_file
    if st.session_state.get("live_video_path") and os.path.exists(st.session_state["live_video_path"]):
        live_video_path = st.session_state["live_video_path"]
        
        # --- Video Analysis Process ---
        st.divider()
        st.markdown("#### ⚙️ Processing Live Feed")
        
        # Track whether we've already processed this exact video file to avoid recomputing on every keystroke
        if "live_frames_video" not in st.session_state or st.session_state.get("last_processed_live_path") != live_video_path:
            with st.status("Analyzing Live Video Content...", expanded=True) as status:
                st.write("🎞️ Extracting frames from live video...")
                frames_live, timestamps_live, fps_live, v_length_live = extract_frames(live_video_path)
                
                st.write("🧠 Generative AI Model generating embeddings for live video...")
                image_embs_live = get_image_embeddings(frames_live)
                
                status.update(label="✅ Live Video Analyzed Successfully!", state="complete", expanded=False)
                
                # Save to session to avoid re-running extract_frames continuously
                st.session_state["live_frames_video"] = frames_live
                st.session_state["live_timestamps_video"] = timestamps_live
                st.session_state["live_fps_video"] = fps_live
                st.session_state["live_v_length_video"] = v_length_live
                st.session_state["live_image_embs_video"] = image_embs_live
                st.session_state["last_processed_live_path"] = live_video_path
        
        frames_live = st.session_state["live_frames_video"]
        timestamps_live = st.session_state["live_timestamps_video"]
        fps_live = st.session_state["live_fps_video"]
        image_embs_live = st.session_state["live_image_embs_video"]
        
        # --- Metrics Section ---
        cols_l = st.columns(4)
        with cols_l[0]:
            st.metric(label="Duration", value=f"{st.session_state['live_v_length_video']:.1f}s")
        with cols_l[1]:
            st.metric(label="Total Frames", value=f"{len(frames_live)}")
        with cols_l[2]:
            st.metric(label="FPS", value=f"{fps_live}")
        with cols_l[3]:
            st.metric(label="Sampling Interval", value=f"{frame_interval}s")

        # Expandable preview of all frames
        with st.expander("🖼️ View Extracted Frame Gallery", expanded=False):
            st.markdown("Browse through all images that the AI analyzed:")
            gallery_cols_live = st.columns(6)
            for idx, img in enumerate(frames_live):
                with gallery_cols_live[idx % 6]:
                    st.image(img, caption=f"{timestamps_live[idx]:.1f}s", use_container_width=True)

        # --- Search Interface ---
        st.divider()
        st.markdown("### 🔎 Search Your Live Recording")
        
        suggestions_tab2 = ["🚶 Person", "💻 Laptop", "📱 Phone", "☕ Coffee cup", "🪑 Chair", "📝 Document"]
        chip_cols_2 = st.columns(len(suggestions_tab2))
        for i, sug in enumerate(suggestions_tab2):
            if chip_cols_2[i].button(sug, key=f"sug_tab2_{i}", use_container_width=True):
                st.session_state["search_query_tab2"] = sug.split(" ", 1)[-1]

        col_input_2, col_submit_2 = st.columns([4, 1])
        with col_input_2:
            query_live = st.text_input("Enter natural language query", key="query_tab2", value=st.session_state.get("search_query_tab2", ""), placeholder="e.g. A white coffee cup...", label_visibility="collapsed")
        with col_submit_2:
            search_pressed_live = st.button("🚀 Search Live Clip", key="search_btn_tab2", type="primary", use_container_width=True)

        if search_pressed_live or query_live:
            st.session_state["search_query_tab2"] = query_live
            with st.spinner(f"🔍 Searching live clip for: **{query_live}**"):
                text_emb_live = get_text_embedding(query_live)
                results_live = find_matches(text_emb_live, image_embs_live, timestamps_live)

            if not results_live:
                st.warning("No matches found in your live recording.")
            else:
                st.success(f"Found {len(results_live)} matching scenes!")
                st.markdown(f"## 🎯 Top Matches for '{query_live}'")

                for rank, (score, time_stamp, idx) in enumerate(results_live, 1):
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'<h4 style="color:#e2e8f0;">🏆 Rank #{rank} <span style="font-size:0.8em; color:#818cf8;">(Match Score: {score:.2f})</span></h4>', unsafe_allow_html=True)
                    st.caption(f"📍 Location in live feed: **{time_stamp:.2f} seconds**")
                    
                    data_col_1, data_col_2 = st.columns(2)
                    
                    with data_col_1:
                        st.markdown("**🖼️ Extracted Keyframe**")
                        st.image(frames_live[idx], use_container_width=True)

                    with data_col_2:
                        st.markdown("**🎬 Contextual Clip**")
                        # Extract context clip specifically from the live video
                        clip_path_live = extract_clip(live_video_path, time_stamp, fps_live)
                        if clip_path_live:
                            st.video(clip_path_live)
                        else:
                            st.info("Failed to extract context clip.")
                    st.markdown('</div>', unsafe_allow_html=True)