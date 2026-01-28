"""
Singleton model loaders for shared resources across pipeline.

This module provides thread-safe initialization of expensive models (EasyOCR, Whisper)
as module-level singletons to avoid redundant loading across multiple workers.
"""

import threading
import easyocr
import whisper
from typing import Optional

# Thread locks for thread-safe singleton initialization
_ocr_lock = threading.Lock()
_whisper_lock = threading.Lock()

# Singleton instances
_ocr_reader: Optional[easyocr.Reader] = None
_whisper_model: Optional[whisper.Whisper] = None


def get_ocr_reader() -> easyocr.Reader:
    """
    Get or initialize the EasyOCR reader singleton.
    
    Thread-safe initialization ensures the model is loaded only once,
    even if multiple threads call this function concurrently.
    
    Returns:
        easyocr.Reader: Shared OCR reader instance
    """
    global _ocr_reader
    
    if _ocr_reader is None:
        with _ocr_lock:
            # Double-check locking pattern
            if _ocr_reader is None:
                print("Initializing EasyOCR reader...")
                _ocr_reader = easyocr.Reader(['en'], gpu=True)
    
    return _ocr_reader


def get_whisper_model() -> whisper.Whisper:
    """
    Get or initialize the Whisper model singleton.
    
    Thread-safe initialization ensures the model is loaded only once,
    even if multiple threads call this function concurrently.
    Uses the 'base' model as a good balance between accuracy and speed.
    
    Returns:
        whisper.Whisper: Shared Whisper model instance
    """
    global _whisper_model
    
    if _whisper_model is None:
        with _whisper_lock:
            # Double-check locking pattern
            if _whisper_model is None:
                print("Initializing Whisper model...")
                _whisper_model = whisper.load_model("base")
    
    return _whisper_model


def preload_models():
    """
    Preload all models to warm up GPU/memory before processing.
    
    Call this once at pipeline start to ensure models are ready.
    Useful for performance profiling and consistent timing.
    """
    print("Preloading models...")
    _ = get_ocr_reader()
    _ = get_whisper_model()
    print("✓ Models preloaded")
