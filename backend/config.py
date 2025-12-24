"""
AudioGhost AI - Centralized Configuration
All settings are configurable via environment variables for Docker/production deployment.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

# Base directories
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent


class Settings:
    """Application settings with environment variable support."""
    
    # ===================
    # Server Configuration
    # ===================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # CORS - comma-separated origins
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000"
    ).split(",")
    
    # ===================
    # Redis Configuration
    # ===================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # ===================
    # File Storage Paths
    # ===================
    # These can be overridden for Docker volume mounts
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
    CHECKPOINTS_DIR: Path = Path(os.getenv("CHECKPOINTS_DIR", str(PROJECT_ROOT / "checkpoints")))
    
    # ===================
    # HuggingFace Configuration
    # ===================
    # Token can be provided via env var (preferred for Docker) or file
    HF_TOKEN: str | None = os.getenv("HF_TOKEN", None)
    # Token file path
    # - Local/dev default: backend/.hf_token (matches existing docs/tools)
    # - Docker: set HF_TOKEN_FILE=/app/data/.hf_token in docker-compose to share between api & worker
    HF_TOKEN_FILE: Path = Path(os.getenv("HF_TOKEN_FILE", str(BASE_DIR / ".hf_token")))
    HF_HOME: str = os.getenv("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
    
    def get_hf_token(self) -> str | None:
        """Get HuggingFace token from env var or file."""
        if self.HF_TOKEN:
            return self.HF_TOKEN
        if self.HF_TOKEN_FILE.exists():
            return self.HF_TOKEN_FILE.read_text().strip()
        return None
    
    def save_hf_token(self, token: str) -> None:
        """Save HuggingFace token to file."""
        self.HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.HF_TOKEN_FILE.write_text(token)
    
    def clear_hf_token(self) -> None:
        """Clear saved HuggingFace token."""
        if self.HF_TOKEN_FILE.exists():
            self.HF_TOKEN_FILE.unlink()
    
    # ===================
    # Device Configuration
    # ===================
    # auto: detect best available (cuda > mps > cpu)
    # cuda: force CUDA (will fail if not available)
    # mps: force Apple Metal (will fail if not available)
    # cpu: force CPU
    DEVICE: Literal["auto", "cuda", "mps", "cpu"] = os.getenv("DEVICE", "auto")
    
    # Default model size for CPU mode (smaller = faster)
    DEFAULT_MODEL_SIZE: str = os.getenv("DEFAULT_MODEL_SIZE", "base")
    CPU_DEFAULT_MODEL_SIZE: str = os.getenv("CPU_DEFAULT_MODEL_SIZE", "small")
    
    # ===================
    # Task Configuration
    # ===================
    TASK_TIME_LIMIT: int = int(os.getenv("TASK_TIME_LIMIT", "3600"))  # 1 hour
    RESULT_EXPIRES: int = int(os.getenv("RESULT_EXPIRES", "86400"))  # 24 hours
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    
    # ===================
    # Processing Defaults
    # ===================
    DEFAULT_CHUNK_DURATION: float = float(os.getenv("DEFAULT_CHUNK_DURATION", "25.0"))
    DEFAULT_USE_FLOAT32: bool = os.getenv("DEFAULT_USE_FLOAT32", "false").lower() == "true"
    
    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = self.HF_HOME


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience export
settings = get_settings()

