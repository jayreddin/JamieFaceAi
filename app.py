import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import onnxruntime as rt
import threading
import mediapipe as mp
import os
from twilio.rest import Client
import cv2
from skimage.transform import SimilarityTransform
from types import SimpleNamespace
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv
from typing import Dict, Any



class Detection(SimpleNamespace):
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None   

class Identity(SimpleNamespace):
    detection: Detection = Detection()
    name: str = None
    embedding: np.ndarray = None
    face: np.ndarray = None


class Match(SimpleNamespace):
    subject_id: Identity = Identity()
    gallery_id: Identity = Identity()
    distance: float = None
    name: str = None


class Grabber(object):
    def __init__(self, video_receiver) -> None:
        self.currentFrame = None
        self.capture = video_receiver
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True

    def update_frame(self) -> None:
        while True:
            self.currentFrame = self.capture.get_frame()

    def get_frame(self) -> av.VideoFrame:
        return self.currentFrame


# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 1.2

# Get twilio ice server configuration using twilio credentials from environment variables (set in streamlit secrets)
# Ref: https://www.twilio.com/docs/stun-turn/api
client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
token = client.tokens.create()
ICE_SERVERS = token.ice_servers

# Set page layout for streamlit to wide
st.set_page_config(
    layout="wide",
    page_title="Live Face Recognition",
    page_icon=":sunglasses:",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Add CSS for mobile optimization
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .stApp {
            max-width: 100vw !important;
        }
        
        /* Improve button sizes on mobile */
        .stButton>button {
            width: 100%;
            height: 3em;
        }
        
        /* Make video container responsive */
        .stVideo {
            width: 100% !important;
            height: auto !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.title("Live Webcam Face Recognition")

st.markdown("**Live Stream**")
ctx_container = st.container()
stream_container = st.empty()

st.markdown("**Matches**")
matches_container = st.info("No matches found yet ...")

st.markdown("**Info**")
info_container = st.empty()


# Init face detector and face recognizer
face_recognizer = rt.InferenceSession("model.fixed.onnx", providers=rt.get_available_providers())
face_detector = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=5,
)


def detect_faces(frame: np.ndarray) -> List[Detection]:
    # Process the frame with the face detector
    result = face_detector.process(frame)

    # Initialize an empty list to store the detected faces
    detections = []

    # Check if any faces were detected
    if result.multi_face_landmarks:
        # Iterate over each detected face
        for count, detection in enumerate(result.multi_face_landmarks):
            # Select 5 Landmarks
            five_landmarks = np.asarray(detection.landmark)[[470, 475, 1, 57, 287]]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = [[landmark.x * frame.shape[1], landmark.y * frame.shape[0]] for landmark in five_landmarks]

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * frame.shape[1] for landmark in detection.landmark]
            all_y_coords = [landmark.y * frame.shape[0] for landmark in detection.landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

            # Create a Detection object for the face
            detection = Detection(
                idx=count,
                bbox=bbox,
                landmarks=landmarks,
                confidence=None,
            )

            # Add the detection to the list
            detections.append(detection)

    # Return the list of detections
    return detections


def recognize_faces(frame: np.ndarray, detections: List[Detection]) -> List[Identity]:
    if not detections:
        return []

    identities = []
    for detection in detections:
        # ALIGNMENT -----------------------------------------------------------
        # Target landmark coordinates (as used in training)
        landmarks_target = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(detection.landmarks, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(frame, tmatrix, (112, 112), borderValue=0.0)
        # ---------------------------------------------------------------------

        # INFERENCE -----------------------------------------------------------
        # Inference face embeddings with onnxruntime
        input_image = (np.asarray([face_aligned]).astype(np.float32) / 255.0).clip(0.0, 1.0)
        embedding = face_recognizer.run(None, {"input_image": input_image})[0][0]
        # ---------------------------------------------------------------------

        # Create Identity object
        identities.append(Identity(detection=detection, embedding=embedding, face=face_aligned))

    return identities


def match_faces(subjects: List[Identity], gallery: List[Identity]) -> List[Match]:
    if len(gallery) == 0 or len(subjects) == 0:
        return []

    # Get Embeddings
    embs_gal = np.asarray([identity.embedding for identity in gallery])
    embs_det = np.asarray([identity.embedding for identity in subjects])

    # Calculate Cosine Distances
    cos_distances = cosine_distances(embs_det, embs_gal)

    # Find Matches
    matches = []
    for ident_idx, identity in enumerate(subjects):
        dists_to_identity = cos_distances[ident_idx]
        idx_min = np.argmin(dists_to_identity)
        if dists_to_identity[idx_min] < SIMILARITY_THRESHOLD:
            matches.append(
                Match(
                    subject_id=identity,
                    gallery_id=gallery[idx_min],
                    distance=dists_to_identity[idx_min],
                )
            )

    # Sort Matches by identity_idx
    matches = sorted(matches, key=lambda match: match.gallery_id.name)

    return matches


def draw_annotations(frame: np.ndarray, detections: List[Detection], matches: List[Match]) -> np.ndarray:
    global timestamp
    shape = np.asarray(frame.shape[:2][::-1])

    # Upscale frame to 1080p for better visualization of drawn annotations
    frame = cv2.resize(frame, (1920, 1080))
    upscale_factor = np.asarray([1920 / shape[0], 1080 / shape[1]])
    shape = np.asarray(frame.shape[:2][::-1])

    # Make frame writeable (for better performance)
    frame.flags.writeable = True

    fps = 1 / (time.time() - timestamp)
    timestamp = time.time()

    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Draw Detections
    for detection in detections:
        # Draw Landmarks
        for landmark in detection.landmarks:
            cv2.circle(
                frame,
                (landmark * upscale_factor).astype(int),
                2,
                (255, 255, 255),
                -1,
            )

        # Draw Bounding Box
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (255, 0, 0),
            2,
        )

        # Draw Index
        cv2.putText(
            frame,
            str(detection.idx),
            (
                ((detection.bbox[1][0] + 2) * upscale_factor[0]).astype(int),
                ((detection.bbox[1][1] + 2) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    # Draw Matches
    for match in matches:
        detection = match.subject_id.detection
        name = match.gallery_id.name

        # Draw Bounding Box in green
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (0, 255, 0),
            2,
        )

        # Draw Banner
        cv2.rectangle(
            frame,
            (
                (detection.bbox[0][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1] - (shape[1] // 25)).astype(int),
            ),
            (
                (detection.bbox[1][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1]).astype(int),
            ),
            (255, 255, 255),
            -1,
        )

        # Draw Name
        cv2.putText(
            frame,
            name,
            (
                ((detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]).astype(int),
                ((detection.bbox[0][1] - shape[1] // 50) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.7,
            (0, 0, 0),
            2,
        )

        # Draw Distance
        cv2.putText(
            frame,
            f" Distance: {match.distance:.2f}",
            (
                ((detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]).astype(int),
                ((detection.bbox[0][1] - shape[1] // 350) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert frame to numpy array
    frame = frame.to_ndarray(format="rgb24")

    # Run face detection
    detections = detect_faces(frame)

    # Run face recognition
    subjects = recognize_faces(frame, detections)

    # Run face matching
    matches = match_faces(subjects, gallery)

    # Draw annotations
    frame = draw_annotations(frame, detections, matches)

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    return frame, matches


# Sidebar for face gallery
with st.sidebar:
    st.markdown("# Face Gallery")
    files = st.sidebar.file_uploader(
        "Upload images to gallery",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Init gallery
    gallery = []
    for file in files:
        # Read file bytes
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

        # Decode image and convert from BGR to RGB
        img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Detect faces
        detections = detect_faces(img)

        if detections:
            # recognize faces
            subjects = recognize_faces(img, detections[:1])

            # Add subjects to gallery
            gallery.append(
                Identity(
                    name=os.path.splitext(file.name)[0],
                    embedding=subjects[0].embedding,
                    face=subjects[0].face,
                )
            )

    # Show gallery images
    st.image(
        image=[identity.face for identity in gallery],
        caption=[identity.name for identity in gallery],
    )

    camera_facing_mode = st.radio(
    "Camera Mode",
    ("user", "environment"),
    horizontal=True,
    help='"user" for front camera, "environment" for back camera'
)

# Start streaming component
with ctx_container:
    ctx = webrtc_streamer(
        key="LiveFaceRecognition",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": ICE_SERVERS},
        media_stream_constraints={
            "video": {
                "width": 1920,
                "facingMode": camera_facing_mode
            },
            "audio": False
        },
        video_frame_callback=video_frame_callback,
    )

# Initialize frame grabber
grabber = Grabber(ctx.video_receiver)

if ctx.state.playing:
    # Start frame grabber in background thread
    grabber.thread.start()
    timestamp = time.time()

    # Start main loop
    while True:
        frame = grabber.get_frame()
        if frame is not None:
            # Print frame timestamp to streamlit
            info_container.write(f"Frame timestamp: {frame.time}")

            # Run face detection and recognition
            frame, matches = video_frame_callback(frame)

            # Convert frame to numpy array
            frame = frame.to_ndarray(format="rgb24")

            # Show Stream
            stream_container.image(frame, channels="RGB")

            # Show Matches
            if matches:
                matches_container.image(
                    image=[match.subject_id.face for match in matches],
                    caption=[match.gallery_id.name for match in matches],
                )
            else:
                matches_container.info("No matches found yet ...")
