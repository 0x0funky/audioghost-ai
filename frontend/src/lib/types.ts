export interface TaskResult {
    original_path: string;
    ghost_path: string;
    clean_path: string;
    description: string;
    mode: string;
    audio_duration?: number;
    processing_time?: number;
    model_size?: string;
    model_variant?: string;
    quality?: string;
    stereo?: boolean;
    precision?: string;
    seed?: number;
    original_filename?: string;
    video_path?: string;
    is_video?: boolean;
}
