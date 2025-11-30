import sys
import os
import argparse
import uvicorn
import shutil
import cv2
import numpy as np
import subprocess
import base64
import requests
import time
import traceback
import logging
import uuid
import signal
import importlib
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Any, Optional, Tuple, Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# ================= 环境配置 =================

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ================= FaceFusion 依赖 =================

import facefusion
from facefusion import state_manager, face_analyser, content_analyser, inference_manager
from facefusion.vision import read_static_image, write_image, read_image
from facefusion.face_detector import detect_faces
from facefusion.face_landmarker import detect_face_landmark

# Face Swapper and Enhancer modules - simplified imports
try:
    # Try to import the core modules using correct paths
    from facefusion.face_swapper import swap_face as face_swapper_core, get_model_options as face_swapper_options
    from facefusion.face_enhancer import enhance_face as face_enhancer_core, get_model_options as face_enhancer_options
    FACE_ENHANCER_AVAILABLE = True
    FACE_SWAPPER_AVAILABLE = True
    print("DEBUG: Successfully imported face processing modules from facefusion.*")
except ImportError as e:
    print(f"WARNING: Could not import advanced face processing modules: {e}")
    # Try alternative imports
    try:
        from facefusion.processors.modules.face_swapper.core import swap_face as face_swapper_core, get_model_options as face_swapper_options
        from facefusion.processors.modules.face_enhancer.core import enhance_face as face_enhancer_core, get_model_options as face_enhancer_options
        FACE_ENHANCER_AVAILABLE = True
        FACE_SWAPPER_AVAILABLE = True
        print("DEBUG: Imported face processing modules from processors path")
    except ImportError as e2:
        print(f"WARNING: Alternative import also failed: {e2}")
        FACE_ENHANCER_AVAILABLE = False
        FACE_SWAPPER_AVAILABLE = False

# ================= Monkey Patch =================

def apply_patches():
    """Apply necessary monkey patches for NSFW bypass and numpy dimension issues"""

    # 1. NSFW Bypass - Force disable content checking
    def no_nsfw(frame):
        return False

    if hasattr(content_analyser, 'detect_nsfw'):
        content_analyser.detect_nsfw = no_nsfw
    else:
        print("WARNING: content_analyser.detect_nsfw not found")

    # 2. Numpy Dimension Fix - Ensure 3-channel BGR format for detector
    original_detect_faces = detect_faces

    def patched_detect_faces(vision_frame):
        if vision_frame is None:
            return [], [], []

        # Ensure numpy array format
        if not isinstance(vision_frame, np.ndarray):
            vision_frame = np.array(vision_frame)

        # Ensure 3D array with 3 channels (BGR)
        if vision_frame.ndim == 2:
            vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_GRAY2BGR)
        elif vision_frame.ndim == 3:
            if vision_frame.shape[2] == 4:
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGRA2BGR)
            elif vision_frame.shape[2] == 1:
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_GRAY2BGR)
            elif vision_frame.shape[2] != 3:
                # Handle unexpected channel counts
                vision_frame = vision_frame[:, :, :3]

        # Ensure uint8 data type
        if vision_frame.dtype != np.uint8:
            vision_frame = (vision_frame * 255).astype(np.uint8) if vision_frame.max() <= 1.0 else vision_frame.astype(np.uint8)

        return original_detect_faces(vision_frame)

    # Monkey patch the function
    import facefusion.face_detector
    facefusion.face_detector.detect_faces = patched_detect_faces

    print(">>> Monkey Patches Applied (Safe Detect & NSFW Bypass)")

# ================= 日志配置 =================

ARGS = None
TASK_STATUS = {}  # In-memory task status store

class StreamToLogger(object):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if "INFO:" in line:
                self.logger.info(line)
            elif "WARNING:" in line:
                self.logger.warning(line)
            elif "ERROR:" in line:
                self.logger.error(line)
            else:
                self.logger.log(self.level, line)
    def flush(self): pass
    def isatty(self): return False

def setup_logging(log_file_path: str):
    """Setup proper logging configuration"""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create console handler for startup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler]
    )

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

# ================= Lifespan Management =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    print(">>> Initializing FaceFusion 3.5 Master Models...")
    try:
        apply_patches()

        # Initialize state with proper settings (CLI compatible)
        state_manager.init_item('execution_providers', ['cuda'])
        state_manager.init_item('execution_device_ids', [0])  # GPU 0
        state_manager.init_item('execution_thread_count', 18)
        state_manager.init_item('execution_queue_count', 1)
        state_manager.init_item('download_providers', ['github', 'huggingface'])
        state_manager.init_item('download_scope', 'full')

        # Initialize additional CLI-compatible states
        state_manager.init_item('command', 'run')  # Simulate CLI mode
        state_manager.init_item('ui_layouts', ['default'])  # Use default layout

        # API server runs in CLI context automatically via detect_app_context()
        import facefusion

        # Face detector settings
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])

        # Face landmarker settings
        state_manager.init_item('face_landmarker_score', 0.5)
        state_manager.init_item('face_landmarker_model', '2dfan4')

        # Face selector settings
        state_manager.init_item('face_selector_mode', 'reference')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('face_selector_gender', 'none')
        state_manager.init_item('face_selector_age_start', None)
        state_manager.init_item('face_selector_age_end', None)
        state_manager.init_item('reference_face_distance', 0.6)

        # Face enhancer settings
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 80)
        state_manager.init_item('face_enhancer_weight', 1.0)

        # Face swapper settings
        state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')
        state_manager.init_item('face_swapper_pixel_boost', '128x128')

        # Face mask settings (required for face swapping)
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', 0.5)  # Higher blur for smoother edges
        state_manager.init_item('face_mask_padding', [5, 5, 5, 5])  # Small padding to avoid hard edges
        state_manager.init_item('face_mask_areas', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Output settings
        state_manager.init_item('output_image_quality', 80)
        state_manager.init_item('output_video_encoder', 'h264_nvenc')
        state_manager.init_item('output_video_quality', 80)
        state_manager.init_item('output_audio_encoder', 'aac')

        # Video processing settings
        state_manager.init_item('video_memory_strategy', 'tolerant')
        state_manager.init_item('temp_frame_format', 'jpeg')

        print(">>> Warming up models...")
        # Create dummy frame for model initialization
        dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)

        # Warm up face analyser
        try:
            get_one_face_from_frame(dummy_frame)
        except Exception as e:
            print(f"WARNING: Face analyser warm-up failed: {e}")

        # Warm up face swapper model explicitly
        try:
            if FACE_SWAPPER_AVAILABLE:
                print("DEBUG: Warming up face swapper model...")
                # Create a dummy source face for testing
                dummy_source_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                dummy_target_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

                # Extract faces for testing
                source_faces = face_analyser.get_many_faces([dummy_source_frame])
                target_faces = face_analyser.get_many_faces([dummy_target_frame])

                if source_faces and target_faces:
                    # Test swap operation
                    test_result = face_swapper_core(
                        source_face=source_faces[0],
                        target_face=target_faces[0],
                        temp_vision_frame=dummy_target_frame
                    )
                    if test_result is not None:
                        print(f"DEBUG: Face swapper warm-up successful, result shape: {test_result.shape}, dtype: {test_result.dtype}, min: {test_result.min()}, max: {test_result.max()}")
                    else:
                        print("WARNING: Face swapper warm-up returned None")
                else:
                    print("WARNING: Could not detect faces for swapper warm-up")
            else:
                print("WARNING: Face swapper not available for warm-up")
        except Exception as e:
            print(f"WARNING: Face swapper warm-up failed: {e}")
            import traceback
            traceback.print_exc()

        # Warm up face enhancer model explicitly
        try:
            if FACE_ENHANCER_AVAILABLE:
                print("DEBUG: Warming up face enhancer model...")
                # Create a dummy frame for testing
                dummy_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

                # Extract face for testing
                faces = face_analyser.get_many_faces([dummy_frame])

                if faces:
                    # Test enhancement operation
                    test_result = face_enhancer_core(
                        target_face=faces[0],
                        temp_vision_frame=dummy_frame
                    )
                    if test_result is not None:
                        print(f"DEBUG: Face enhancer warm-up successful, result shape: {test_result.shape}, dtype: {test_result.dtype}, min: {test_result.min()}, max: {test_result.max()}")
                    else:
                        print("WARNING: Face enhancer warm-up returned None")
                else:
                    print("WARNING: Could not detect face for enhancer warm-up")
            else:
                print("WARNING: Face enhancer not available for warm-up")
        except Exception as e:
            print(f"WARNING: Face enhancer warm-up failed: {e}")
            import traceback
            traceback.print_exc()

        print(">>> Service Ready.")
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)

    yield

    print(">>> Shutting down...")
    # Clean up inference manager
    try:
        inference_manager.clear_inference_pool()
    except:
        pass

app = FastAPI(lifespan=lifespan, title="FaceFusion API", version="3.5")

# ================= 工具函数 =================

def get_temp_file_path(filename: str) -> str:
    """Get temporary file path"""
    temp_root = os.path.join(ARGS.output_dir, "temp")
    os.makedirs(temp_root, exist_ok=True)
    return os.path.join(temp_root, filename)

def standard_response(code: int = 200, message: str = "success", data: Any = None):
    """Standard API response format"""
    if data is None: data = {}
    return JSONResponse(content={"code": code, "message": message, "data": data})

def download_file(url: str, save_path: str, timeout: int = 60):
    """Download file from URL"""
    print(f"DEBUG: Starting download: {url} -> {save_path}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # 禁用SSL验证警告
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        with requests.get(url, stream=True, timeout=timeout, headers=headers, verify=False) as r:
            r.raise_for_status()

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                total_size = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉保持连接的新块
                        f.write(chunk)
                        total_size += len(chunk)

                print(f"DEBUG: Wrote {total_size} bytes to {save_path}")

        # 验证文件是否成功创建
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"DEBUG: Download success: {save_path}, size: {file_size} bytes")
            if file_size == 0:
                raise Exception("Downloaded file is empty")
            elif file_size != total_size:
                print(f"WARNING: File size mismatch: expected {total_size}, got {file_size}")
        else:
            raise Exception("File was not created after download")

    except requests.exceptions.Timeout as e:
        print(f"ERROR: Timeout downloading {url}: {e}")
        raise Exception(f"Download timeout [{url}]: {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection error downloading {url}: {e}")
        raise Exception(f"Connection failed [{url}]: {e}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: HTTP error downloading {url}: {e}")
        raise Exception(f"HTTP error [{url}]: {e}")
    except IOError as e:
        print(f"ERROR: File write error saving to {save_path}: {e}")
        raise Exception(f"File write error [{save_path}]: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error downloading {url}: {e}")
        raise Exception(f"Download failed [{url}]: {e}")

def save_base64(b64_str: str, save_path: str):
    """Save base64 data to file"""
    try:
        # Remove data URL prefix if present
        if "," in b64_str: b64_str = b64_str.split(",")[1]
        with open(save_path, 'wb') as f:
            f.write(base64.b64decode(b64_str))
    except Exception as e:
        raise Exception(f"Base64 decode failed: {e}")

def generate_cover(media_path: str) -> str:
    """Generate cover image from media file"""
    if not os.path.exists(media_path): return ""
    cover_path = os.path.splitext(media_path)[0] + "_cover.jpg"

    try:
        if media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # For images, copy directly
            shutil.copy(media_path, cover_path)
        elif media_path.lower().endswith(('.mp4', '.mov', '.avi', '.gif')):
            # For videos and GIFs, extract first frame
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(cover_path, frame)
            cap.release()
        return cover_path if os.path.exists(cover_path) else ""
    except Exception as e:
        print(f"Failed to generate cover: {e}")
        return ""

def get_output_info(task_id: str, url: str) -> Tuple[str, str]:
    """Get output file path and media type"""
    now = datetime.now()
    sub_dir = os.path.join(ARGS.output_dir, f"{now.year}/{now.month:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    # Clean URL and determine extension
    clean_url = url.split("?")[0]
    ext = os.path.splitext(clean_url)[1].lower()

    # Map extensions to output formats
    if ext == '.gif':
        return os.path.join(sub_dir, f"{task_id}.gif"), 'gif'
    elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        return os.path.join(sub_dir, f"{task_id}.mp4"), 'video'
    else:
        return os.path.join(sub_dir, f"{task_id}.jpg"), 'image'

def calc_process_params(cap: cv2.VideoCapture, req_fps: Optional[float], req_min_side: Optional[int]) -> Tuple[float, tuple]:
    """Calculate processing parameters (FPS and resolution)"""
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate final FPS
    final_fps = src_fps
    if req_fps is not None and req_fps > 0:
        if req_fps < src_fps:
            final_fps = req_fps

    # Calculate final resolution
    final_res = (src_w, src_h)
    src_min = min(src_w, src_h)
    if req_min_side is not None and req_min_side > 0:
        if req_min_side < src_min:
            scale = req_min_side / src_min
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            # Ensure even dimensions
            if new_w % 2 != 0: new_w -= 1
            if new_h % 2 != 0: new_h -= 1
            final_res = (new_w, new_h)

    return final_fps, final_res

def force_bgr(frame):
    """Ensure frame is in BGR format with uint8 dtype"""
    if frame is None:
        return None
    if frame.size == 0:
        return None

    # Convert to uint8 if needed
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # Convert color format if needed
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3:
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            return frame
    return frame

def get_one_face_from_frame(frame):
    """Convenience function to get the first face from a single frame"""
    if frame is None:
        return None
    faces = face_analyser.get_many_faces([frame])
    if not faces:
        return None
    return face_analyser.get_one_face(faces, 0)

# ================= 核心处理逻辑 =================

async def process_swap_task_async(task_id: str, params: dict):
    """Async wrapper for swap task processing"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, process_swap_task, task_id, params)

def process_frame_core(frame: np.ndarray, faces: List, loaded_pairs: List[dict], mode: str) -> np.ndarray:
    """Process frame with face swapping and enhancement"""
    if not faces:
        return frame

    print(f"DEBUG: Input frame - shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
    print(f"DEBUG: Found {len(faces)} faces to process")

    # Ensure frame has correct format and dtype
    frame = force_bgr(frame)
    original_dtype = frame.dtype

    print(f"DEBUG: After force_bgr - shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

    # 1. Face Swapping
    for i, face in enumerate(faces):
        target_src = None
        if mode == "single":
            target_src = loaded_pairs[0]["source_face"]
            print(f"DEBUG: Face {i} - Using single mode, source face available: {target_src is not None}")
        else:
            # Multi-face: find matching reference face
            for p in loaded_pairs:
                if p["ref_embedding"] is not None:
                    similarity = np.dot(face.embedding, p["ref_embedding"])
                    print(f"DEBUG: Face {i} - Similarity with reference: {similarity}")
                    if similarity > 0.4:
                        target_src = p["source_face"]
                        break

        if target_src and FACE_SWAPPER_AVAILABLE and face_swapper_core is not None:
            try:
                # Validate face landmarks before processing
                has_face_landmarks = (hasattr(face, 'landmark_set') and face.landmark_set)
                has_src_landmarks = (hasattr(target_src, 'landmark_set') and target_src.landmark_set)

                print(f"DEBUG: Face {i} - Landmarks available - face: {has_face_landmarks}, src: {has_src_landmarks}")

                if has_face_landmarks and has_src_landmarks:

                    print(f"DEBUG: Face {i} - Starting face swap...")
                    # Use the core swap function from face_swapper module
                    # Correct parameters: source_face, target_face, temp_vision_frame
                    swapped_frame = face_swapper_core(
                        source_face=target_src,
                        target_face=face,
                        temp_vision_frame=frame
                    )

                    print(f"DEBUG: Face {i} - Swap result - shape: {swapped_frame.shape if swapped_frame is not None else None}, dtype: {swapped_frame.dtype if swapped_frame is not None else None}")

                    # Validate result before using it
                    if swapped_frame is not None and swapped_frame.size > 0:
                        # Check for NaN or Inf values
                        has_nan = np.any(np.isnan(swapped_frame))
                        has_inf = np.any(np.isinf(swapped_frame))

                        print(f"DEBUG: Face {i} - Validation - NaN: {has_nan}, Inf: {has_inf}, min: {swapped_frame.min()}, max: {swapped_frame.max()}")

                        if not has_nan and not has_inf:
                            frame = swapped_frame
                            print(f"DEBUG: Face {i} - Swap successful, frame updated")
                        else:
                            print(f"WARNING: Face {i} - Invalid values detected in swapped frame, skipping")
                    else:
                        print(f"WARNING: Face {i} - Invalid swapped frame returned, skipping")

                else:
                    print(f"WARNING: Face {i} - Face landmarks not available for swapping")
            except Exception as e:
                print(f"WARNING: Face {i} - Face swap failed: {e}")
                import traceback
                traceback.print_exc()

    # 2. Face Enhancement
    if FACE_ENHANCER_AVAILABLE and face_enhancer_core is not None:
        print(f"DEBUG: Starting face enhancement for {len(faces)} faces")
        try:
            for i, face in enumerate(faces):
                try:
                    # Validate face landmarks before enhancement
                    has_face_landmarks = (hasattr(face, 'landmark_set') and face.landmark_set)

                    print(f"DEBUG: Enhancement - Face {i} has landmarks: {has_face_landmarks}")

                    if has_face_landmarks:
                        print(f"DEBUG: Enhancement - Face {i} - Starting enhancement...")
                        # Correct parameters: target_face, temp_vision_frame
                        enhanced_frame = face_enhancer_core(
                            target_face=face,
                            temp_vision_frame=frame
                        )

                        print(f"DEBUG: Enhancement - Face {i} - Enhanced frame - shape: {enhanced_frame.shape if enhanced_frame is not None else None}")

                        # Validate result before using it
                        if enhanced_frame is not None and enhanced_frame.size > 0:
                            # Check for NaN or Inf values
                            has_nan = np.any(np.isnan(enhanced_frame))
                            has_inf = np.any(np.isinf(enhanced_frame))

                            print(f"DEBUG: Enhancement - Face {i} - Validation - NaN: {has_nan}, Inf: {has_inf}, min: {enhanced_frame.min()}, max: {enhanced_frame.max()}")

                            if not has_nan and not has_inf:
                                frame = enhanced_frame
                                print(f"DEBUG: Enhancement - Face {i} - Enhancement successful")
                            else:
                                print(f"WARNING: Enhancement - Face {i} - Invalid values detected in enhanced frame, skipping")
                        else:
                            print(f"WARNING: Enhancement - Face {i} - Invalid enhanced frame returned, skipping")
                    else:
                        print(f"WARNING: Enhancement - Face {i} - Face landmarks not available for enhancement")
                except Exception as e:
                    print(f"WARNING: Enhancement - Face {i} - Face enhancement failed: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"WARNING: Face enhancement failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("WARNING: face_enhancer_core function not available")

    # Ensure output frame has correct dtype and no invalid values
    if frame is not None:
        # Check for invalid values before fixing
        has_nan = np.any(np.isnan(frame))
        has_inf = np.any(np.isinf(frame))

        print(f"DEBUG: Final frame before cleanup - shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
        print(f"DEBUG: Final frame has NaN: {has_nan}, has Inf: {has_inf}")

        # Fix any potential NaN or Inf values
        frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0)

        # Ensure correct data type
        if frame.dtype != original_dtype:
            print(f"DEBUG: Converting dtype from {frame.dtype} to {original_dtype}")
            frame = frame.astype(original_dtype)

        print(f"DEBUG: Final frame after cleanup - shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

    return frame

def process_swap_task(task_id: str, params: dict):
    """Main swap task processing function"""
    temp_files = []
    output_path = params["output_path"]

    # Update task status
    TASK_STATUS[task_id] = {"status": "processing", "message": "Starting task", "progress": 0}

    try:
        print(f"[{task_id}] Start. Mode: {params['mode']}")

        # Download target file
        target_url = params["target_url"]
        ext = os.path.splitext(target_url.split("?")[0])[1]
        if not ext: ext = ".tmp"
        target_local = get_temp_file_path(f"{task_id}_target{ext}")
        download_file(target_url, target_local)
        temp_files.append(target_local)

        TASK_STATUS[task_id] = {"status": "processing", "message": "Loading faces", "progress": 20}

        # Load source and reference faces
        loaded_pairs = []

        def add_pair(src_url: str, ref_b64: Optional[str] = None, is_single: bool = False):
            """Add a source face pair"""
            try:
                # Download source image
                s_path = get_temp_file_path(f"{task_id}_src_{len(loaded_pairs)}.jpg")
                print(f"DEBUG: Downloading source from {src_url} to {s_path}")
                download_file(src_url, s_path)
                temp_files.append(s_path)

                # 验证文件下载是否成功
                if not os.path.exists(s_path):
                    print(f"ERROR: Downloaded file does not exist: {s_path}")
                    return

                file_size = os.path.getsize(s_path)
                print(f"DEBUG: Downloaded file size: {file_size} bytes")
                if file_size == 0:
                    print(f"ERROR: Downloaded file is empty: {s_path}")
                    return

                # Read source image and extract face
                s_frame = force_bgr(read_static_image(s_path))
                if s_frame is None:
                    print(f"WARNING: Cannot read source image: {s_path}")
                    # 尝试使用 cv2 直接读取
                    try:
                        import cv2
                        s_frame = cv2.imread(s_path)
                        if s_frame is not None:
                            print(f"DEBUG: cv2.imread succeeded, shape: {s_frame.shape}")
                            s_frame = force_bgr(s_frame)
                        else:
                            print(f"ERROR: cv2.imread also failed for: {s_path}")
                            return
                    except Exception as cv_error:
                        print(f"ERROR: cv2.imread fallback failed: {cv_error}")
                        return

                s_face = get_one_face_from_frame(s_frame)
                if not s_face:
                    print(f"WARNING: No face found in source: {s_path}")
                    return

                print(f"DEBUG: Successfully extracted face from: {src_url}")

                # Extract reference face embedding if provided
                ref_emb = None
                if not is_single and ref_b64:
                    r_path = get_temp_file_path(f"{task_id}_ref_{len(loaded_pairs)}.jpg")
                    save_base64(ref_b64, r_path)
                    temp_files.append(r_path)

                    r_frame = force_bgr(read_static_image(r_path))
                    if r_frame is not None:
                        r_face = get_one_face_from_frame(r_frame)
                        if r_face:
                            ref_emb = r_face.embedding

                loaded_pairs.append({
                    "source_face": s_face,
                    "ref_embedding": ref_emb,
                    "source_url": src_url
                })

            except Exception as e:
                print(f"WARNING: Failed to add face pair: {e}")

        # Load face pairs based on mode
        if params["mode"] == "single":
            add_pair(params["face_url"], is_single=True)
        else:
            for pair in params["pairs"]:
                add_pair(pair["source"], pair["reference"])

        if not loaded_pairs:
            raise Exception("No valid source faces found.")

        print(f"[{task_id}] Loaded {len(loaded_pairs)} source face(s)")
        TASK_STATUS[task_id] = {"status": "processing", "message": "Processing media", "progress": 40}

        media_type = params["media_type"]

        # Process Image
        if media_type == 'image':
            frame = force_bgr(read_image(target_local))
            if frame is None:
                raise Exception("Failed to load target image")

            # Apply resolution scaling if requested
            req_min = params.get("min_resolution")
            if req_min:
                h, w = frame.shape[:2]
                min_side = min(h, w)
                if req_min < min_side:
                    scale = req_min / min_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Detect faces and process
            frame = force_bgr(frame)
            faces = face_analyser.get_many_faces([frame])
            if faces:
                TASK_STATUS[task_id] = {"status": "processing", "message": "Swapping faces", "progress": 60}
                frame = process_frame_core(frame, faces, loaded_pairs, params["mode"])

            # Save result
            write_image(output_path, frame)

        # Process Video/GIF
        else:
            cap = cv2.VideoCapture(target_local)
            final_fps, final_res = calc_process_params(
                cap, params.get("fps"), params.get("min_resolution")
            )

            # Set default FPS
            if media_type == 'gif' and (final_fps <= 0 or final_fps > 60):
                final_fps = 15
            if media_type == 'video' and (final_fps <= 0 or final_fps > 120):
                final_fps = 30

            # Create temporary output video
            temp_out = get_temp_file_path(f"{task_id}_proc.mp4")
            temp_files.append(temp_out)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, final_fps, final_res)

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Resize if needed
                if (frame.shape[1], frame.shape[0]) != final_res:
                    frame = cv2.resize(frame, final_res, interpolation=cv2.INTER_AREA)

                # Convert to BGR
                frame = force_bgr(frame)

                # Process faces
                faces = face_analyser.get_many_faces([frame])
                print(f"DEBUG: Frame {frame_count} - Detected {len(faces)} faces")

                if faces:
                    original_frame = frame.copy()
                    processed_frame = process_frame_core(frame, faces, loaded_pairs, params["mode"])

                    # Check if processing actually changed the frame
                    if processed_frame is not None:
                        # Calculate difference between original and processed
                        diff = np.abs(processed_frame.astype(np.float32) - original_frame.astype(np.float32))
                        mean_diff = diff.mean()
                        max_diff = diff.max()

                        print(f"DEBUG: Frame {frame_count} - Frame difference - Mean: {mean_diff:.2f}, Max: {max_diff}")

                        # Analyze pixel value distribution
                        orig_mean = original_frame.mean()
                        orig_std = original_frame.std()
                        proc_mean = processed_frame.mean()
                        proc_std = processed_frame.std()

                        print(f"DEBUG: Frame {frame_count} - Original - Mean: {orig_mean:.2f}, Std: {orig_std:.2f}")
                        print(f"DEBUG: Frame {frame_count} - Processed - Mean: {proc_mean:.2f}, Std: {proc_std:.2f}")

                        # Check if processed frame is just black
                        non_zero_ratio = np.count_nonzero(processed_frame) / processed_frame.size
                        zero_ratio = np.count_nonzero(processed_frame == 0) / processed_frame.size
                        high_ratio = np.count_nonzero(processed_frame > 240) / processed_frame.size

                        print(f"DEBUG: Frame {frame_count} - Non-zero pixel ratio: {non_zero_ratio:.4f}")
                        print(f"DEBUG: Frame {frame_count} - Zero pixel ratio: {zero_ratio:.4f}")
                        print(f"DEBUG: Frame {frame_count} - High pixel (>240) ratio: {high_ratio:.4f}")

                        # Check if face region looks valid
                        if zero_ratio > 0.5:  # More than 50% black pixels
                            print(f"WARNING: Frame {frame_count} - Processed frame is mostly black ({zero_ratio:.4f}% zero), using original")
                            frame = original_frame
                        elif high_ratio > 0.8:  # More than 80% bright pixels
                            print(f"WARNING: Frame {frame_count} - Processed frame has too many bright pixels ({high_ratio:.4f}% >240), using original")
                            frame = original_frame
                        elif proc_std < 10:  # Very low variation (likely uniform color)
                            print(f"WARNING: Frame {frame_count} - Processed frame has low variation (std: {proc_std:.2f}), using original")
                            frame = original_frame
                        else:
                            print(f"DEBUG: Frame {frame_count} - Frame looks valid, using result")
                            frame = processed_frame

                        if non_zero_ratio < 0.1:  # Less than 10% non-zero pixels
                            print(f"WARNING: Frame {frame_count} - Processed frame is mostly black ({non_zero_ratio:.4f}% non-zero)")
                            # Use original frame if processed is mostly black
                            frame = original_frame
                        else:
                            frame = processed_frame
                    else:
                        print(f"WARNING: Frame {frame_count} - process_frame_core returned None")
                else:
                    print(f"DEBUG: Frame {frame_count} - No faces detected, keeping original")

                out.write(frame)
                frame_count += 1

                # Update progress
                if total_frames > 0 and frame_count % 50 == 0:
                    progress = min(80, 40 + int((frame_count / total_frames) * 40))
                    TASK_STATUS[task_id] = {
                        "status": "processing",
                        "message": f"Processing frame {frame_count}/{total_frames}",
                        "progress": progress
                    }

            cap.release()
            out.release()

            TASK_STATUS[task_id] = {"status": "processing", "message": "Finalizing output", "progress": 90}

            # Convert to final format
            if media_type == 'gif':
                # Convert to GIF using ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_out,
                    '-vf', 'fps=15,scale=480:-1:flags=lanczos',
                    output_path
                ], check=True, capture_output=True)
            else:
                # Convert to MP4 with audio if available
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', temp_out
                ]

                # Add audio if available in source
                try:
                    cmd.extend(['-i', target_local, '-c:v', 'copy', '-c:a', 'aac'])
                    cmd.extend(['-map', '0:v:0', '-map', '1:a:0', '-shortest'])
                except:
                    cmd.extend(['-c:v', 'copy'])

                cmd.append(output_path)

                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    # Fallback: just move the processed video
                    shutil.move(temp_out, output_path)

        # Generate cover
        cover_path = generate_cover(output_path)

        # Update final status
        TASK_STATUS[task_id] = {
            "status": "success",
            "message": "Task completed successfully",
            "progress": 100,
            "result": output_path,
            "cover": cover_path
        }

        print(f"[{task_id}] Success: {output_path}")

    except Exception as e:
        err_msg = str(e)
        print(f"[{task_id}] FAILED: {err_msg}")
        traceback.print_exc()

        # Save error info
        with open(output_path + ".error", "w") as f:
            f.write(err_msg)

        # Update task status
        TASK_STATUS[task_id] = {
            "status": "failed",
            "message": f"Task failed: {err_msg}",
            "progress": 0
        }

    finally:
        # Clean up temporary files
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

# ================= API Request Models =================

class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")

class SingleSwapRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    target_file: str = Field(..., description="URL to target media file")
    face_file: str = Field(..., description="URL to source face image")
    fps: Optional[float] = Field(None, description="Target FPS (optional)")
    min_resolution: Optional[int] = Field(None, description="Minimum resolution side (optional)")

class MultiSwapPair(BaseModel):
    reference: str = Field(..., description="Base64 encoded reference face")
    source: str = Field(..., description="URL to source face image")

class MultiSwapRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    target_file: str = Field(..., description="URL to target media file")
    swap_pairs: List[MultiSwapPair] = Field(..., description="List of face swap pairs")
    fps: Optional[float] = Field(None, description="Target FPS (optional)")
    min_resolution: Optional[int] = Field(None, description="Minimum resolution side (optional)")

# ================= API Endpoints =================

@app.get("/api/docs")
async def get_api_docs():
    """API documentation"""
    docs = [
        {
            "name": "Detect Face",
            "url": "/api/detect",
            "method": "POST",
            "description": "Detect faces in a base64 image.",
            "input": {"image_base64": "string (Base64)"},
            "output": {"faces": [{"bbox": [0,0,0,0], "score": 0.9, "landmarks": [[0,0], [0,0]]}]}
        },
        {
            "name": "Single Face Swap",
            "url": "/api/swap/single",
            "method": "POST",
            "description": "Swap all faces with one source face + Enhance.",
            "input": {
                "task_id": "string", "target_file": "string (URL)", "face_file": "string (URL)",
                "fps": "float", "min_resolution": "int"
            },
            "output": {"task_id": "string", "status": "processing"}
        },
        {
            "name": "Multi Face Swap",
            "url": "/api/swap/multi",
            "method": "POST",
            "description": "Swap specific faces + Enhance.",
            "input": {
                "task_id": "string", "target_file": "string (URL)",
                "swap_pairs": [{"reference": "string (Base64)", "source": "string (URL)"}],
                "fps": "float", "min_resolution": "int"
            },
            "output": {"task_id": "string", "status": "processing"}
        },
        {
            "name": "Check Status",
            "url": "/api/status/{task_id}",
            "method": "GET",
            "description": "Get task result.",
            "output": {"status": "success", "result": "/path/to/file", "cover": "/path/to/cover"}
        }
    ]
    return standard_response(data=docs)

@app.get("/gradio_api/app_id")
async def gradio_app_id():
    """Gradio compatibility endpoint"""
    return JSONResponse(content={
        "app_id": "facefusion-api",
        "version": "3.5"
    })

@app.post("/api/detect")
async def detect_face(req: DetectRequest):
    """Detect faces in a base64 encoded image"""
    tmp_name = get_temp_file_path(f"detect_{uuid.uuid4()}.jpg")
    try:
        # Decode base64 image
        if "," in req.image_base64:
            b64_data = req.image_base64.split(",")[1]
        else:
            b64_data = req.image_base64

        img_bytes = base64.b64decode(b64_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return standard_response(code=400, message="Invalid image data")

        # Detect faces - pass frame as list according to API
        faces = face_analyser.get_many_faces([frame])

        if faces is None:
            results = []
        else:
            results = []
            for f in faces:
                face_data = {
                    "bbox": f.bounding_box.tolist() if hasattr(f.bounding_box, 'tolist') else list(f.bounding_box),
                    "score": float(f.score_set.get('detector', 0.0)) if f.score_set else 0.0,
                    "angle": float(f.angle) if hasattr(f, 'angle') else 0.0
                }
                # Add landmarks if available
                if hasattr(f, 'landmark_set') and f.landmark_set:
                    if '5' in f.landmark_set:
                        landmarks = f.landmark_set['5']
                        if hasattr(landmarks, 'tolist'):
                            landmarks = landmarks.tolist()
                        elif isinstance(landmarks, np.ndarray):
                            landmarks = landmarks.astype(float).tolist()
                        face_data["landmarks"] = landmarks
                results.append(face_data)

        return standard_response(data={"faces": results, "count": len(results)})

    except Exception as e:
        traceback.print_exc()
        return standard_response(code=500, message=f"Face detection failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except:
                pass

@app.post("/api/swap/single")
async def swap_single(req: SingleSwapRequest, bg_tasks: BackgroundTasks):
    """Single face swap endpoint"""
    try:
        # Validate inputs
        if not req.task_id:
            return standard_response(code=400, message="Task ID is required")
        if not req.target_file:
            return standard_response(code=400, message="Target file URL is required")
        if not req.face_file:
            return standard_response(code=400, message="Source face file URL is required")

        # Get output path
        path, m_type = get_output_info(req.task_id, req.target_file)

        # Prepare parameters
        params = {
            "mode": "single",
            "target_url": req.target_file,
            "face_url": req.face_file,
            "output_path": path,
            "media_type": m_type,
            "fps": req.fps,
            "min_resolution": req.min_resolution
        }

        # Add to background tasks
        bg_tasks.add_task(process_swap_task_async, req.task_id, params)

        return standard_response(data={"task_id": req.task_id, "status": "processing"})

    except Exception as e:
        return standard_response(code=500, message=f"Single swap failed: {str(e)}")

@app.post("/api/swap/multi")
async def swap_multi(req: MultiSwapRequest, bg_tasks: BackgroundTasks):
    """Multi-face swap endpoint"""
    try:
        # Validate inputs
        if not req.task_id:
            return standard_response(code=400, message="Task ID is required")
        if not req.target_file:
            return standard_response(code=400, message="Target file URL is required")
        if not req.swap_pairs or len(req.swap_pairs) == 0:
            return standard_response(code=400, message="At least one swap pair is required")

        # Get output path
        path, m_type = get_output_info(req.task_id, req.target_file)

        # Prepare parameters
        params = {
            "mode": "multi",
            "target_url": req.target_file,
            "pairs": [{"reference": p.reference, "source": p.source} for p in req.swap_pairs],
            "output_path": path,
            "media_type": m_type,
            "fps": req.fps,
            "min_resolution": req.min_resolution
        }

        # Add to background tasks
        bg_tasks.add_task(process_swap_task_async, req.task_id, params)

        return standard_response(data={"task_id": req.task_id, "status": "processing"})

    except Exception as e:
        return standard_response(code=500, message=f"Multi swap failed: {str(e)}")

@app.get("/api/status/{task_id}")
async def check_status(task_id: str):
    """Check task status"""
    try:
        # Check in-memory status first
        if task_id in TASK_STATUS:
            status_info = TASK_STATUS[task_id]
            if status_info["status"] in ["success", "failed"]:
                return standard_response(data=status_info)
            else:
                return standard_response(data={"task_id": task_id, "status": "processing", **status_info})

        # Check filesystem for completed tasks
        now = datetime.now()
        full_dir = os.path.join(ARGS.output_dir, f"{now.year}/{now.month:02d}")

        if not os.path.exists(full_dir):
            return standard_response(data={"task_id": task_id, "status": "processing", "message": "Task not found in memory"})

        # Look for result files
        target_file = None
        error_file = None

        for f in os.listdir(full_dir):
            if f.startswith(task_id):
                if f.endswith(".error"):
                    error_file = f
                elif "_cover.jpg" not in f and f.endswith(('.mp4', '.jpg', '.gif')):
                    target_file = f

        # Check for error
        if error_file:
            error_path = os.path.join(full_dir, error_file)
            try:
                with open(error_path, 'r') as f:
                    error_msg = f.read().strip()
                return standard_response(
                    code=500,
                    message=f"Task failed: {error_msg}",
                    data={"task_id": task_id, "status": "failed"}
                )
            except:
                return standard_response(
                    code=500,
                    message="Task failed with unknown error",
                    data={"task_id": task_id, "status": "failed"}
                )

        # Check for result
        if target_file:
            result_path = os.path.join(full_dir, target_file)
            cover_path = generate_cover(result_path)

            return standard_response(data={
                "task_id": task_id,
                "status": "success",
                "result": result_path,
                "cover": cover_path,
                "message": "Task completed successfully"
            })

        # Still processing
        return standard_response(data={"task_id": task_id, "status": "processing", "message": "Task is still processing"})

    except Exception as e:
        traceback.print_exc()
        return standard_response(code=500, message=f"Status check failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return standard_response(data={"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceFusion 3.5 Master API Server")
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log_file', type=str, required=True, help='Log file path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')

    ARGS = parser.parse_args()

    # Setup logging
    setup_logging(ARGS.log_file)

    print(f">>> Starting FaceFusion API Server on {ARGS.host}:{ARGS.port}")
    print(f">>> Output directory: {ARGS.output_dir}")
    print(f">>> Log file: {ARGS.log_file}")

    # Run server
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_config=None)
