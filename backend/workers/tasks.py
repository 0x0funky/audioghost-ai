"""
Celery Tasks - Audio Separation Workers
With SAM Audio Lite optimization for low VRAM usage
"""
import os
import sys
import gc
from pathlib import Path
from typing import Optional, List

from celery import current_task

from workers.celery_app import celery_app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

# Use configured output directory
OUTPUT_DIR = settings.OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)


def get_best_device() -> str:
    """
    Detect the best available device based on settings and hardware.
    
    Priority: CUDA > MPS > CPU (unless overridden by DEVICE setting)
    """
    import torch
    
    device_setting = settings.DEVICE.lower()
    
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available")
    
    elif device_setting == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available")
    
    elif device_setting == "cpu":
        return "cpu"
    
    else:  # auto
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

# Global model cache to avoid reloading
_model_cache = {}
_processor_cache = {}


def update_progress(progress: int, message: str):
    """Update task progress"""
    current_task.update_state(
        state="PROGRESS",
        meta={"progress": progress, "message": message}
    )


def create_lite_model(model_name: str, hf_token: str = None):
    """
    Create a memory-optimized SAM Audio model by removing unused components.
    
    Reduces VRAM usage from ~11GB to ~4-5GB by:
    - Replacing vision_encoder with a dummy
    - Disabling visual_ranker
    - Disabling text_ranker
    - Disabling span_predictor
    """
    import torch
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    print(f"Loading {model_name} (lite mode)...")
    
    # Load model
    if hf_token:
        model = SAMAudio.from_pretrained(model_name, token=hf_token)
    else:
        model = SAMAudio.from_pretrained(model_name)
    
    processor = SAMAudioProcessor.from_pretrained(model_name)
    
    print("Optimizing model for low VRAM...")
    
    # Get vision encoder dim before deleting
    vision_dim = model.vision_encoder.dim if hasattr(model.vision_encoder, 'dim') else 1024
    
    # Delete heavy components
    del model.vision_encoder
    gc.collect()
    
    # Store the dim for _get_video_features
    model._vision_encoder_dim = vision_dim
    
    # Replace _get_video_features to not use vision_encoder
    def _get_video_features_lite(self, video, audio_features):
        B, T, _ = audio_features.shape
        return audio_features.new_zeros(B, self._vision_encoder_dim, T)
    
    import types
    model._get_video_features = types.MethodType(_get_video_features_lite, model)
    
    # Delete rankers
    if hasattr(model, 'visual_ranker') and model.visual_ranker is not None:
        del model.visual_ranker
        model.visual_ranker = None
        gc.collect()
    
    if hasattr(model, 'text_ranker') and model.text_ranker is not None:
        del model.text_ranker
        model.text_ranker = None
        gc.collect()
    
    # Delete span predictor
    if hasattr(model, 'span_predictor') and model.span_predictor is not None:
        del model.span_predictor
        model.span_predictor = None
        gc.collect()
    
    if hasattr(model, 'span_predictor_transform') and model.span_predictor_transform is not None:
        del model.span_predictor_transform
        model.span_predictor_transform = None
        gc.collect()
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Model optimization complete!")
    
    return model, processor


def get_or_load_lite_model(model_name: str, hf_token: str, device: str, dtype):
    """Get cached lite model or create it - only keeps ONE model in memory"""
    import torch
    
    # Include dtype in cache key to ensure correct model is loaded
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
    cache_key = f"{model_name}_lite_{device}_{dtype_str}"
    
    print(f"[DEBUG] Looking for cached model with key: {cache_key}")
    print(f"[DEBUG] Current cache keys: {list(_model_cache.keys())}")
    
    if cache_key not in _model_cache:
        print(f"[DEBUG] Cache miss - creating new lite model")
        
        # IMPORTANT: Clear any existing models first to free memory
        if len(_model_cache) > 0:
            print(f"[DEBUG] Clearing {len(_model_cache)} existing model(s) from cache...")
            for old_key in list(_model_cache.keys()):
                del _model_cache[old_key]
            for old_key in list(_processor_cache.keys()):
                del _processor_cache[old_key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[DEBUG] GPU Memory after clearing old models: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        model, processor = create_lite_model(model_name, hf_token)
        
        print(f"[DEBUG] Converting model to {device} with dtype {dtype}")
        model = model.eval().to(device, dtype)
        
        _model_cache[cache_key] = model
        _processor_cache[model_name] = processor
        
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print(f"[DEBUG] Cache hit - using existing model")
    
    return _model_cache[cache_key], _processor_cache[model_name]




def cleanup_gpu_memory():
    """Clean up GPU memory after task"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def load_audio_any_format(audio_path: str):
    """
    Load audio from a file path without relying on torchcodec.

    - First tries torchaudio.load() (uses available backend, e.g. soundfile).
    - If that fails (common for mp3 when only soundfile backend is available),
      falls back to pydub+ffmpeg.

    Returns:
        (waveform, sample_rate) where waveform is torch.Tensor [channels, time]
    """
    import torch
    import torchaudio

    try:
        return torchaudio.load(audio_path)
    except Exception as e:
        print(f"[WARN] torchaudio.load failed for '{audio_path}': {e}")
        print("[WARN] Falling back to pydub/ffmpeg decoder...")

        from pydub import AudioSegment  # uses ffmpeg binary
        import numpy as np

        seg = AudioSegment.from_file(audio_path)
        sample_rate = int(seg.frame_rate)

        # Convert to mono to match downstream pipeline
        seg = seg.set_channels(1)

        sample_width = int(seg.sample_width)  # bytes per sample
        samples = np.array(seg.get_array_of_samples())

        # Normalize to [-1, 1] float32
        if sample_width == 1:
            # 8-bit audio is typically unsigned
            samples = samples.astype(np.int16) - 128
            max_val = 128.0
        else:
            max_val = float(1 << (8 * sample_width - 1))

        waveform = torch.from_numpy(samples.astype(np.float32) / max_val).unsqueeze(0)
        return waveform, sample_rate


@celery_app.task(bind=True)
def separate_audio_task(
    self,
    audio_path: str,
    description: str,
    mode: str = "extract",
    anchors: Optional[List] = None,
    model_size: str = "base",
    chunk_duration: float = 25.0,
    use_float32: bool = False
):
    """
    Separate audio using SAM Audio Lite (memory optimized)
    
    Args:
        audio_path: Path to input audio file
        description: Text prompt for separation
        mode: "extract" or "remove"
        anchors: Optional temporal anchors [["+", start, end], ...]
        model_size: Model size (small/base/large)
        chunk_duration: Audio chunk duration in seconds (5-60)
        use_float32: Use float32 precision for better quality
    
    Returns:
        Dictionary with paths to output files
    """
    import torch
    import torchaudio
    import time
    from huggingface_hub import login
    
    task_id = self.request.id
    device = get_best_device()
    print(f"[INFO] Using device: {device}")
    
    # Debug: Show received parameter
    print(f"[DEBUG] use_float32 parameter received: {use_float32} (type: {type(use_float32).__name__})")
    
    # Set precision based on use_float32 parameter
    if use_float32 or device == "cpu":
        dtype = torch.float32
        print(f"[DEBUG] Using float32 precision (High Quality Mode)")
    else:
        dtype = torch.bfloat16
        print(f"[DEBUG] Using bfloat16 precision (Memory Optimized)")
    
    # Start timing
    start_time = time.time()
    
    try:
        update_progress(5, "Initializing...")
        
        # Load HuggingFace token from config (supports env var or file)
        hf_token = settings.get_hf_token()
        if not hf_token:
            raise Exception("HuggingFace token not found. Please authenticate first or set HF_TOKEN environment variable.")
        login(token=hf_token)
        
        # Select model based on size
        model_name = f"facebook/sam-audio-{model_size}"
        
        update_progress(10, f"Loading {model_name} (lite mode)...")
        
        # Clean up before loading
        cleanup_gpu_memory()
        
        # Load lite model (with caching)
        model, processor = get_or_load_lite_model(model_name, hf_token, device, dtype)
        
        update_progress(30, "Loading audio...")
        
        # Get sample rate
        sample_rate = processor.audio_sampling_rate
        
        # Load and preprocess audio
        audio, orig_sr = load_audio_any_format(audio_path)
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Calculate audio duration
        audio_duration = audio.shape[1] / sample_rate
        print(f"[DEBUG] Audio duration: {audio_duration:.2f}s")
        
        # Chunking settings (from parameter, clamped to 5-60)
        CHUNK_DURATION = max(5.0, min(60.0, chunk_duration))
        MAX_CHUNK_SAMPLES = int(sample_rate * CHUNK_DURATION)
        
        # Check if chunking is needed
        if audio.shape[1] > MAX_CHUNK_SAMPLES:
            print(f"[DEBUG] Audio is {audio_duration:.1f}s, using chunking ({CHUNK_DURATION}s chunks)")
            
            # Split audio into chunks
            audio_tensor = audio.squeeze(0).to(device, dtype)
            chunks = torch.split(audio_tensor, MAX_CHUNK_SAMPLES, dim=-1)
            total_chunks = len(chunks)
            
            out_target = []
            out_residual = []
            
            for i, chunk in enumerate(chunks):
                # Update progress
                chunk_progress = 30 + int((i / total_chunks) * 50)
                update_progress(chunk_progress, f"Processing chunk {i+1}/{total_chunks}...")
                
                # Skip very short chunks
                if chunk.shape[-1] < sample_rate:  # Less than 1 second
                    print(f"[DEBUG] Skipping chunk {i+1} (too short)")
                    continue
                
                # Prepare batch for this chunk
                batch = processor(
                    audios=[chunk.unsqueeze(0)],
                    descriptions=[description]
                ).to(device)
                
                # Run separation
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        result = model.separate(
                            batch,
                            predict_spans=False,
                            reranking_candidates=1
                        )
                
                out_target.append(result.target[0].cpu())
                out_residual.append(result.residual[0].cpu())
                
                # Clean up chunk results
                del batch, result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all chunks
            target_audio = torch.cat(out_target, dim=-1).clamp(-1, 1).float().unsqueeze(0)
            residual_audio = torch.cat(out_residual, dim=-1).clamp(-1, 1).float().unsqueeze(0)
            
            del out_target, out_residual, chunks, audio_tensor
            
        else:
            print(f"[DEBUG] Audio is {audio_duration:.1f}s, processing as single batch")
            
            update_progress(50, "Running separation...")
            
            # Process entire audio at once
            # IMPORTANT: Pass the already-loaded audio tensor instead of a file path.
            # This avoids SAM-Audio's internal loader path that can depend on torchcodec.
            audio_tensor = audio.to(device, dtype)
            batch = processor(
                audios=[audio_tensor],
                descriptions=[description]
            ).to(device)
            
            # Run separation
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    result = model.separate(
                        batch,
                        predict_spans=False,
                        reranking_candidates=1
                    )
            
            target_audio = result.target[0].float().unsqueeze(0).cpu()
            residual_audio = result.residual[0].float().unsqueeze(0).cpu()
            
            del batch, result
        
        update_progress(80, "Saving results...")
        
        # Output paths
        output_base = OUTPUT_DIR / task_id
        original_path = output_base.with_suffix(".original.wav")
        ghost_path = output_base.with_suffix(".ghost.wav")
        clean_path = output_base.with_suffix(".clean.wav")
        
        # Save original audio
        torchaudio.save(str(original_path), audio.cpu(), sample_rate)
        
        # Save separated audio
        if mode == "extract":
            torchaudio.save(str(ghost_path), target_audio, sample_rate)
            torchaudio.save(str(clean_path), residual_audio, sample_rate)
        else:
            torchaudio.save(str(ghost_path), target_audio, sample_rate)
            torchaudio.save(str(clean_path), residual_audio, sample_rate)
        
        update_progress(100, "Complete!")
        
        # Aggressive cleanup
        print(f"[DEBUG] Cleaning up GPU memory...")
        del target_audio, residual_audio, audio
        
        gc.collect()
        cleanup_gpu_memory()
        
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"[DEBUG] Processing completed in {processing_time:.2f}s for {audio_duration:.2f}s audio")
        
        return {
            "original_path": str(original_path),
            "ghost_path": str(ghost_path),
            "clean_path": str(clean_path),
            "description": description,
            "mode": mode,
            "audio_duration": round(audio_duration, 2),
            "processing_time": round(processing_time, 2),
            "model_size": model_size
        }

        
    except Exception as e:
        gc.collect()
        cleanup_gpu_memory()
        raise Exception(f"Separation failed: {str(e)}")



@celery_app.task(bind=True)
def match_pattern_task(
    self,
    audio_path: str,
    sample_path: str,
    threshold: float = 0.85,
    model_size: str = "base"
):
    """
    Find and remove sounds similar to a sample
    
    Args:
        audio_path: Path to input audio file
        sample_path: Path to sample audio file
        threshold: Similarity threshold (0-1)
        model_size: Model size (small/base/large)
    
    Returns:
        Dictionary with paths to output files and matched segments
    """
    # TODO: Implement pattern matching with CLAP embeddings
    # This is a placeholder for MVP v1.0
    
    update_progress(50, "Pattern matching not yet implemented in MVP")
    
    return {
        "status": "not_implemented",
        "message": "Pattern matching will be available in v1.1"
    }
