"""
Test Video Only - Simple video-based audio separation using SAM Audio
Uses small model with bfloat16 for lower memory usage
"""
import torch
import torchaudio
import gc

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Load model (small for lower memory)
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    model_name = "facebook/sam-audio-base"
    print(f"Loading {model_name}...")
    
    model = SAMAudio.from_pretrained(model_name).to(device, dtype).eval()
    processor = SAMAudioProcessor.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Video file
    video_file = "office.mp4"
    description = "walking sound"
    
    print(f"\nProcessing video: {video_file}")
    print(f"Description: '{description}'")
    
    # Process
    inputs = processor(audios=[video_file], descriptions=[description]).to(device)
    
    print("Running separation...")
    if torch.cuda.is_available():
        print(f"GPU Memory before separation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
        result = model.separate(inputs)
    
    if torch.cuda.is_available():
        print(f"GPU Memory after separation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Save results
    sample_rate = processor.audio_sampling_rate
    
    target_audio = result.target[0].float().unsqueeze(0).cpu()
    residual_audio = result.residual[0].float().unsqueeze(0).cpu()
    
    torchaudio.save("video_target.wav", target_audio, sample_rate)
    torchaudio.save("video_residual.wav", residual_audio, sample_rate)
    
    print("\nDone!")
    print("- video_target.wav: Extracted audio (target)")
    print("- video_residual.wav: Remaining audio (residual)")
    
    # Cleanup
    del model, processor, inputs, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
