"""
SAM Audio Lite - Lightweight version for low VRAM usage (4-6GB)
Disables vision_encoder, rankers, and span_predictor to reduce memory
WITH CHUNKING - Same logic as Celery worker for memory comparison
"""
import torch
import gc
import time


def show_gpu_memory(label: str = ""):
    """Show complete GPU memory stats (matches nvidia-smi more closely)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[GPU Memory{' - ' + label if label else ''}] "
              f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")


def create_lite_model(model_name: str = "facebook/sam-audio-base", token: str = None):
    """
    Create a memory-optimized SAM Audio model by removing unused components.
    
    This can reduce VRAM usage from ~11GB to ~4GB by:
    - Replacing vision_encoder with a dummy (saves ~2GB)
    - Disabling visual_ranker (saves ~2GB)
    - Disabling text_ranker (saves ~2GB)
    - Disabling span_predictor (saves ~1-2GB)
    
    Returns:
        model: Optimized SAM Audio model
        processor: SAM Audio processor
    """
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    print(f"Loading {model_name}...")
    
    # Load model
    if token:
        model = SAMAudio.from_pretrained(model_name, token=token)
    else:
        model = SAMAudio.from_pretrained(model_name)
    
    processor = SAMAudioProcessor.from_pretrained(model_name)
    
    print("Optimizing model for low VRAM...")
    
    # Get vision encoder dim before deleting
    vision_dim = model.vision_encoder.dim if hasattr(model.vision_encoder, 'dim') else 1024
    
    # Delete heavy components
    del model.vision_encoder
    gc.collect()
    print("  - Removed vision_encoder")
    
    # Store the dim for _get_video_features
    model._vision_encoder_dim = vision_dim
    
    # Replace _get_video_features to not use vision_encoder
    def _get_video_features_lite(self, video, audio_features):
        B, T, _ = audio_features.shape
        # Always return zeros since we're not using video
        return audio_features.new_zeros(B, self._vision_encoder_dim, T)
    
    # Bind the new method
    import types
    model._get_video_features = types.MethodType(_get_video_features_lite, model)
    
    # Delete rankers
    if hasattr(model, 'visual_ranker') and model.visual_ranker is not None:
        del model.visual_ranker
        model.visual_ranker = None
        gc.collect()
        print("  - Removed visual_ranker")
    
    if hasattr(model, 'text_ranker') and model.text_ranker is not None:
        del model.text_ranker
        model.text_ranker = None
        gc.collect()
        print("  - Removed text_ranker")
    
    # Delete span predictor
    if hasattr(model, 'span_predictor') and model.span_predictor is not None:
        del model.span_predictor
        model.span_predictor = None
        gc.collect()
        print("  - Removed span_predictor")
    
    if hasattr(model, 'span_predictor_transform') and model.span_predictor_transform is not None:
        del model.span_predictor_transform
        model.span_predictor_transform = None
        gc.collect()
        print("  - Removed span_predictor_transform")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Model optimization complete!")
    
    return model, processor


if __name__ == "__main__":
    import torchaudio
    from pathlib import Path
    
    # =====================================
    # CONFIGURATION
    # =====================================
    USE_BF16 = True  # Set to False to use float32 (better quality, more VRAM)
    # =====================================
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"USE_BF16: {USE_BF16}")
    
    # Clear GPU memory and reset peak stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    show_gpu_memory("Before loading model")
    
    # Create lite model
    model, processor = create_lite_model()
    model = model.eval().to(device, dtype)
    
    show_gpu_memory("After loading model")
    
    # Test audio
    test_audio = "test_chunk.mp3"
    description = "singing voice"
    
    if not Path(test_audio).exists():
        print(f"\nNo test file at {test_audio}, creating test tone...")
        sample_rate = 16000
        duration = 10
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * torch.sin(2 * 3.14159 * 200 * t)
        audio = audio.unsqueeze(0)
        test_audio = "test_audio.wav"
        torchaudio.save(test_audio, audio, sample_rate)
    
    print(f"\nProcessing: {test_audio}")
    print(f"Description: '{description}'")
    
    # Start timing
    start_time = time.time()
    
    # Load audio
    sample_rate = processor.audio_sampling_rate
    audio, orig_sr = torchaudio.load(test_audio)
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Calculate audio duration
    audio_duration = audio.shape[1] / sample_rate
    print(f"\nAudio duration: {audio_duration:.2f}s")
    
    # ===============================
    # CHUNKING LOGIC (same as worker)
    # ===============================
    CHUNK_DURATION = 25.0  # 25 seconds per chunk
    MAX_CHUNK_SAMPLES = int(sample_rate * CHUNK_DURATION)
    
    if audio.shape[1] > MAX_CHUNK_SAMPLES:
        print(f"Audio is {audio_duration:.1f}s, using chunking ({CHUNK_DURATION}s chunks)")
        
        # Split audio into chunks
        audio_tensor = audio.squeeze(0).to(device, dtype)
        chunks = torch.split(audio_tensor, MAX_CHUNK_SAMPLES, dim=-1)
        total_chunks = len(chunks)
        
        out_target = []
        out_residual = []
        
        show_gpu_memory("Before chunked separation")
        
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{total_chunks}...")
            
            # Skip very short chunks
            if chunk.shape[-1] < sample_rate:  # Less than 1 second
                print(f"  Skipping chunk {i+1} (too short: {chunk.shape[-1]/sample_rate:.2f}s)")
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
            
            show_gpu_memory(f"After chunk {i+1}")
            
            # Clean up chunk results
            del batch, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        target_audio = torch.cat(out_target, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        residual_audio = torch.cat(out_residual, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        
        del out_target, out_residual, chunks, audio_tensor
        
    else:
        print(f"Audio is {audio_duration:.1f}s, processing as single batch (no chunking needed)")
        
        # Prepare inputs
        print("\nPreparing batch...")
        batch = processor(
            audios=[test_audio],
            descriptions=[description],
        ).to(device)
        
        # Run separation
        print("Running separation...")
        show_gpu_memory("Before separation")
        
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
            result = model.separate(
                batch, 
                predict_spans=False,
                reranking_candidates=1
            )
        
        show_gpu_memory("After separation")
        
        target_audio = result.target[0].float().unsqueeze(0).cpu()
        residual_audio = result.residual[0].float().unsqueeze(0).cpu()
        
        del batch, result
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Save results
    print("\n" + "="*50)
    print("Saving results...")
    
    torchaudio.save("output_target.wav", target_audio, sample_rate)
    torchaudio.save("output_residual.wav", residual_audio, sample_rate)
    
    # Cleanup
    del target_audio, residual_audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    show_gpu_memory("After cleanup")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Audio duration:   {audio_duration:.2f}s ({audio_duration/60:.2f} min)")
    print(f"Processing time:  {processing_time:.2f}s")
    print(f"Speed:           {audio_duration/processing_time:.2f}x realtime")
    print(f"Chunking:        {'Yes (' + str(len(chunks) if 'chunks' in dir() else '?') + ' chunks)' if audio_duration > CHUNK_DURATION else 'No (single batch)'}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        print(f"Peak GPU Memory: {peak_gb:.2f}GB (reserved: {reserved_gb:.2f}GB)")
    print("="*60)
    print("Output files:")
    print("  - output_target.wav: Extracted audio")
    print("  - output_residual.wav: Residual audio")
