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
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Any, Optional, Tuple
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ================= 环境配置 =================

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ================= FaceFusion 依赖 =================

import facefusion
from facefusion import state_manager, face_analyser
from facefusion.vision import read_static_image, write_image, read_image
import facefusion.face_detector
import facefusion.content_analyser

try:
    from facefusion.processors.modules import face_swapper
except ImportError:
    print("CRITICAL: face_swapper module not found.")
    sys.exit(1)

try:
    from facefusion.processors.modules import face_enhancer
except ImportError:
    face_enhancer = None
    print("WARNING: face_enhancer module not found.")

# ================= Monkey Patch =================

def apply_patches():
    print(">>> Applying Monkey Patches...")
    
    # 1. 定义补丁函数：强制维度检查
    # 获取原始的检测函数引用，防止递归死锁
    original_detect_faces = facefusion.face_detector.detect_faces

    def patched_detect_faces(vision_frame):
        if vision_frame is None:
            return [], [], []
        
        # 强制转为 numpy 数组
        if not isinstance(vision_frame, np.ndarray):
            vision_frame = np.array(vision_frame)

        # --- 核心修复：维度地狱 ---
        # 情况 A: 2维数组 (H, W) -> 灰度图，转 BGR
        if vision_frame.ndim == 2:
            vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_GRAY2BGR)
        # 情况 B: 3维数组，但通道不对
        elif vision_frame.ndim == 3:
            if vision_frame.shape[2] == 4: # BGRA -> BGR
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGRA2BGR)
            elif vision_frame.shape[2] == 1: # (H, W, 1) -> BGR
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_GRAY2BGR)
        
        # 确保是 uint8 类型，防止 float 类型导致 opencv 报错
        if vision_frame.dtype != np.uint8:
            vision_frame = vision_frame.astype(np.uint8)

        return original_detect_faces(vision_frame)

    # 2. 应用补丁 (Patching)
    
    # [Fix A] 修改定义源头
    facefusion.face_detector.detect_faces = patched_detect_faces
    
    # [Fix B] CRITICAL: 修改引用者 (face_analyser)
    # 因为 face_analyser 使用了 'from ... import ...'，必须覆盖它的局部引用
    if hasattr(facefusion.face_analyser, 'detect_faces'):
        facefusion.face_analyser.detect_faces = patched_detect_faces
        print(">>> Patched: facefusion.face_analyser.detect_faces")
        
    # [Fix C] 防止其他潜在引用 (防御性编程)
    if hasattr(facefusion.content_analyser, 'detect_faces'):
        facefusion.content_analyser.detect_faces = patched_detect_faces

    # 3. NSFW 屏蔽补丁
    def no_nsfw(frame): return False
    if hasattr(facefusion, 'content_analyser'):
        if hasattr(facefusion.content_analyser, 'detect_nsfw'):
            facefusion.content_analyser.detect_nsfw = no_nsfw
            print(">>> Patched: NSFW Check Disabled")
    
    print(">>> Monkey Patches Applied Successfully.")

# ================= 日志配置 =================

ARGS = None

class StreamToLogger(object):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if "INFO:" in line: self.logger.info(line)
            else: self.logger.log(self.level, line)
    def flush(self): pass
    def isatty(self): return False

def setup_logging(log_file_path: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path, filemode='a'
    )
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.INFO)

# ================= Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> Initializing FaceFusion 3.5 Master Models...")
    try:
        apply_patches()

        state_manager.init_item('execution_providers', ['cuda'])
        state_manager.init_item('execution_thread_count', 4)
        state_manager.init_item('execution_queue_count', 1)
        
        state_manager.init_item('face_detector_model', 'yoloface') 
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        state_manager.init_item('face_detector_angles', [0, 90, 180, 270])
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])
        
        state_manager.init_item('face_landmarker_score', 0.5)
        state_manager.init_item('face_selector_mode', 'reference')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('face_selector_gender', 'none')
        state_manager.init_item('face_selector_age_start', None)
        state_manager.init_item('face_selector_age_end', None)
        state_manager.init_item('reference_face_distance', 0.6)
        
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 80)
        state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')
        state_manager.init_item('face_swapper_pixel_boost', '128x128')
        
        state_manager.init_item('output_image_quality', 80)
        state_manager.init_item('output_video_encoder', 'libx264')
        state_manager.init_item('output_video_quality', 80)
        state_manager.init_item('output_audio_encoder', 'aac')

        print(">>> Warming up models...")
        dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        try: face_analyser.get_one_face(dummy_frame)
        except: pass 

        if hasattr(face_swapper, 'pre_check'): face_swapper.pre_check()
        if face_enhancer and hasattr(face_enhancer, 'pre_check'): face_enhancer.pre_check()
        
        print(">>> Service Ready.")
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
    yield
    print(">>> Shutting down...")

app = FastAPI(lifespan=lifespan)

# ================= 工具函数 =================

def get_temp_file_path(filename: str) -> str:
    temp_root = os.path.join(ARGS.output_dir, "temp")
    os.makedirs(temp_root, exist_ok=True)
    return os.path.join(temp_root, filename)

def standard_response(code: int = 200, message: str = "success", data: Any = None):
    if data is None: data = {}
    return JSONResponse(content={"code": code, "message": message, "data": data})

def download_file(url: str, save_path: str):
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Download Failed [{url}]: {e}")
        raise e

def save_base64(b64_str: str, save_path: str):
    try:
        if "," in b64_str: b64_str = b64_str.split(",")[1]
        with open(save_path, 'wb') as f:
            f.write(base64.b64decode(b64_str))
    except Exception as e:
        print(f"Base64 Decode Failed: {e}")
        raise e

def generate_cover(media_path: str) -> str:
    if not os.path.exists(media_path): return ""
    cover_path = os.path.splitext(media_path)[0] + "_cover.jpg"
    try:
        if media_path.lower().endswith(('.jpg', '.jpeg')):
            shutil.copy(media_path, cover_path)
        elif media_path.lower().endswith(('.mp4', '.gif', '.mov', '.avi')):
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()
            if ret: cv2.imwrite(cover_path, frame)
            cap.release()
        else:
            img = cv2.imread(media_path)
            if img is not None: cv2.imwrite(cover_path, img)
        return cover_path
    except: return ""

def get_output_info(task_id: str, url: str) -> Tuple[str, str]:
    now = datetime.now()
    sub_dir = os.path.join(ARGS.output_dir, f"{now.year}/{now.month:02d}")
    os.makedirs(sub_dir, exist_ok=True)
    clean_url = url.split("?")[0]
    ext = os.path.splitext(clean_url)[1].lower()
    if ext == '.gif': return os.path.join(sub_dir, f"{task_id}.gif"), 'gif'
    elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']: return os.path.join(sub_dir, f"{task_id}.mp4"), 'video'
    else: return os.path.join(sub_dir, f"{task_id}.jpg"), 'image'

def calc_process_params(cap: cv2.VideoCapture, req_fps: Optional[float], req_min_side: Optional[int]) -> Tuple[float, tuple]:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final_fps = src_fps
    if req_fps is not None and req_fps > 0:
        if req_fps < src_fps: final_fps = req_fps
    final_res = (src_w, src_h)
    src_min = min(src_w, src_h)
    if req_min_side is not None and req_min_side > 0:
        if req_min_side < src_min:
            scale = req_min_side / src_min
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            if new_w % 2 != 0: new_w -= 1
            if new_h % 2 != 0: new_h -= 1
            final_res = (new_w, new_h)
    return final_fps, final_res

def force_bgr(frame):
    if frame is None: return None
    if frame.size == 0: return None
    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
    if frame.ndim == 2: return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4: return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1: return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

# ================= 核心处理逻辑 =================

def process_frame_core(frame, faces, loaded_pairs, mode):
    # 1. Swap
    for face in faces:
        target_src = None
        if mode == "single":
            target_src = loaded_pairs[0]["source_face"]
        else:
            for p in loaded_pairs:
                if p["ref_embedding"] is not None and np.dot(face.embedding, p["ref_embedding"]) > 0.4:
                    target_src = p["source_face"]
                    break
        if target_src:
            frame = face_swapper.swap_face(frame, target_src, face)
    
    # 2. Enhance
    if face_enhancer:
        for face in faces:
             frame = face_enhancer.enhance_face(frame, face)
    return frame

def process_swap_task(task_id: str, params: dict):
    temp_files = []
    output_path = params["output_path"]
    try:
        print(f"[{task_id}] Start. Mode: {params['mode']}")
        target_url = params["target_url"]
        ext = os.path.splitext(target_url.split("?")[0])[1]
        if not ext: ext = ".tmp"
        target_local = get_temp_file_path(f"{task_id}_target{ext}")
        download_file(target_url, target_local)
        temp_files.append(target_local)

        # Load Faces
        loaded_pairs = [] 
        def add_pair(src_url, ref_b64=None, is_single=False):
            s_path = get_temp_file_path(f"{task_id}_src_{len(loaded_pairs)}.jpg")
            download_file(src_url, s_path)
            temp_files.append(s_path)
            
            s_frame = force_bgr(read_static_image(s_path))
            if s_frame is None: raise Exception(f"Cannot read source image: {s_path}")
            s_face = face_analyser.get_one_face(s_frame)
            if not s_face:
                print(f"[{task_id}] Warn: No face in {s_path}")
                return
            ref_emb = None
            if not is_single and ref_b64:
                r_path = get_temp_file_path(f"{task_id}_ref_{len(loaded_pairs)}.jpg")
                save_base64(ref_b64, r_path)
                temp_files.append(r_path)
                r_frame = force_bgr(read_static_image(r_path))
                if r_frame is not None:
                    r_face = face_analyser.get_one_face(r_frame)
                    if r_face: ref_emb = r_face.embedding
            loaded_pairs.append({"source_face": s_face, "ref_embedding": ref_emb})

        if params["mode"] == "single": add_pair(params["face_url"], is_single=True)
        else:
            for p in params["pairs"]: add_pair(p["source"], p["reference"])

        if not loaded_pairs: raise Exception("No valid source faces found.")

        media_type = params["media_type"]
        
        # Process Image
        if media_type == 'image':
            frame = force_bgr(read_image(target_local))
            if frame is None: raise Exception("Failed to load target image")
            
            req_min = params.get("min_resolution")
            if req_min:
                h, w = frame.shape[:2]
                min_side = min(h, w)
                if req_min < min_side:
                    scale = req_min / min_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            frame = force_bgr(frame)
            faces = face_analyser.get_many_faces(frame)
            if faces:
                frame = process_frame_core(frame, faces, loaded_pairs, params["mode"])
            write_image(output_path, frame)

        # Process Video/GIF
        else:
            cap = cv2.VideoCapture(target_local)
            final_fps, final_res = calc_process_params(cap, params.get("fps"), params.get("min_resolution"))
            if media_type == 'gif' and (final_fps <= 0 or final_fps > 60): final_fps = 15
            if media_type == 'video' and (final_fps <= 0 or final_fps > 120): final_fps = 30

            temp_out = get_temp_file_path(f"{task_id}_proc.mp4")
            temp_files.append(temp_out)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, final_fps, final_res)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if (frame.shape[1], frame.shape[0]) != final_res:
                    frame = cv2.resize(frame, final_res, interpolation=cv2.INTER_AREA)
                
                frame = force_bgr(frame)
                faces = face_analyser.get_many_faces(frame)
                if faces:
                     frame = process_frame_core(frame, faces, loaded_pairs, params["mode"])
                out.write(frame)
                frame_idx += 1
                if frame_idx % 50 == 0: print(f"[{task_id}] Frame {frame_idx}")
            cap.release()
            out.release()

            if media_type == 'gif':
                subprocess.run(['ffmpeg', '-y', '-i', temp_out, output_path], check=True)
            else:
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', temp_out, '-i', target_local,
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-map', '0:v:0', '-map', '1:a:0', output_path
                ]
                try: subprocess.run(cmd, check=True)
                except: shutil.move(temp_out, output_path)

        print(f"[{task_id}] Success: {output_path}")

    except Exception as e:
        err_msg = str(e)
        print(f"[{task_id}] FAILED: {err_msg}")
        traceback.print_exc()
        with open(output_path + ".error", "w") as f: f.write(err_msg)
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

# ================= API Endpoints =================

class DetectRequest(BaseModel): image_base64: str
class SingleSwapRequest(BaseModel): 
    task_id: str; target_file: str; face_file: str
    fps: Optional[float] = None; min_resolution: Optional[int] = None
class MultiSwapPair(BaseModel): reference: str; source: str
class MultiSwapRequest(BaseModel): 
    task_id: str; target_file: str; swap_pairs: List[MultiSwapPair]
    fps: Optional[float] = None; min_resolution: Optional[int] = None

@app.get("/api/docs")
async def get_api_docs():
    docs = [
        {
            "name": "Detect Face", "url": "/api/detect", "method": "POST",
            "description": "Detect faces in a base64 image.",
            "input": {"image_base64": "string (Base64)"},
            "output": {"faces": [{"bbox": [0,0,0,0], "score": 0.9}]}
        },
        {
            "name": "Single Face Swap", "url": "/api/swap/single", "method": "POST",
            "description": "Swap all faces with one source face + Enhance.",
            "input": {
                "task_id": "string", "target_file": "string (URL)", "face_file": "string (URL)",
                "fps": "float", "min_resolution": "int"
            },
            "output": {"task_id": "string", "status": "processing"}
        },
        {
            "name": "Multi Face Swap", "url": "/api/swap/multi", "method": "POST",
            "description": "Swap specific faces + Enhance.",
            "input": {
                "task_id": "string", "target_file": "string (URL)",
                "swap_pairs": [{"reference": "string (Base64)", "source": "string (URL)"}],
                "fps": "float", "min_resolution": "int"
            },
            "output": {"task_id": "string", "status": "processing"}
        },
        {
            "name": "Check Status", "url": "/api/status/", "method": "GET",
            "description": "Get task result.",
            "output": {"status": "success", "result": "/path/to/file", "cover": "/path/to/cover"}
        }
    ]
    return standard_response(data=docs)

@app.post("/api/detect")
async def detect_face(req: DetectRequest):
    tmp_name = get_temp_file_path(f"detect_{uuid.uuid4()}.jpg")
    try:
        if "," in req.image_base64: b64_data = req.image_base64.split(",")[1]
        else: b64_data = req.image_base64
        
        img_bytes = base64.b64decode(b64_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None: return standard_response(code=400, message="Invalid image")
        
        faces = face_analyser.get_many_faces(frame)
        
        if faces is None: results = []
        else: results = [{"bbox": f.bbox.tolist(), "score": float(f.score)} for f in faces]
        return standard_response(data={"faces": results})
    except Exception as e:
        traceback.print_exc()
        return standard_response(code=500, message=str(e))

@app.post("/api/swap/single")
async def swap_single(req: SingleSwapRequest, bg_tasks: BackgroundTasks):
    try:
        path, m_type = get_output_info(req.task_id, req.target_file)
        params = {
            "mode": "single", "target_url": req.target_file, 
            "face_url": req.face_file, "output_path": path, 
            "media_type": m_type, "fps": req.fps, 
            "min_resolution": req.min_resolution
        }
        bg_tasks.add_task(process_swap_task, req.task_id, params)
        return standard_response(data={"task_id": req.task_id, "status": "processing"})
    except Exception as e: return standard_response(code=500, message=str(e))

@app.post("/api/swap/multi")
async def swap_multi(req: MultiSwapRequest, bg_tasks: BackgroundTasks):
    try:
        path, m_type = get_output_info(req.task_id, req.target_file)
        params = {
            "mode": "multi", "target_url": req.target_file, 
            "pairs": [{"reference": p.reference, "source": p.source} for p in req.swap_pairs], 
            "output_path": path, "media_type": m_type,
            "fps": req.fps, "min_resolution": req.min_resolution
        }
        bg_tasks.add_task(process_swap_task, req.task_id, params)
        return standard_response(data={"task_id": req.task_id, "status": "processing"})
    except Exception as e: return standard_response(code=500, message=str(e))

@app.get("/api/status/")
async def check_status(task_id: str):
    try:
        now = datetime.now()
        full_dir = os.path.join(ARGS.output_dir, f"{now.year}/{now.month:02d}")
        if not os.path.exists(full_dir): 
            return standard_response(data={"task_id": task_id, "status": "processing"})
        target, error = None, None
        for f in os.listdir(full_dir):
            if f.startswith(task_id):
                if f.endswith(".error"): error = f
                elif "_cover.jpg" not in f and f.endswith(('.mp4', '.jpg', '.gif')): target = f
        if error:
            with open(os.path.join(full_dir, error)) as f: msg = f.read()
            return standard_response(code=500, message=f"Failed: {msg}", data={"task_id": task_id, "status": "failed"})
        if target:
            path = os.path.join(full_dir, target)
            return standard_response(data={"task_id": task_id, "status": "success", "result": path, "cover": generate_cover(path)})
        return standard_response(data={"task_id": task_id, "status": "processing"})
    except Exception as e:
        traceback.print_exc()
        return standard_response(code=500, message=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    ARGS = parser.parse_args()
    setup_logging(ARGS.log_file)
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
