"""
AudioProcessor - Real-time audio capture and playback for voice assistants.

This module provides a reusable AudioProcessor class for handling bidirectional
audio streaming with the Azure VoiceLive SDK.

Usage:
    from audio_processor import AudioProcessor
    
    # Create processor with VoiceLive connection
    processor = AudioProcessor(connection)
    processor.start_playback()
    processor.start_capture()
    
    # Queue audio for playback
    processor.queue_audio(audio_bytes)
    
    # Clean up when done
    processor.shutdown()
"""

from __future__ import annotations
import asyncio
import base64
import logging
import queue
from typing import Optional, TYPE_CHECKING

import pyaudio

if TYPE_CHECKING:
    from azure.ai.voicelive.aio import VoiceLiveConnection

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles real-time audio capture and playback for voice assistants.
    
    This class manages bidirectional audio streaming using PyAudio:
    - Captures audio from microphone and sends to VoiceLive
    - Receives audio from VoiceLive and plays through speakers
    
    Threading Architecture:
    - Main thread: Event loop and UI
    - Capture thread: PyAudio input stream reading (callback-based)
    - Playback thread: PyAudio output stream writing (callback-based)
    
    Audio Format:
    - PCM16 (16-bit signed integer)
    - 24kHz sample rate
    - Mono channel
    """
    
    loop: asyncio.AbstractEventLoop
    
    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection: "VoiceLiveConnection"):
        """
        Initialize the AudioProcessor.
        
        Args:
            connection: VoiceLive connection for sending captured audio
        """
        self.connection = connection
        self.audio = pyaudio.PyAudio()

        # Audio configuration - PCM16, 24kHz, mono
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk_size = 1200  # 50ms of audio at 24kHz

        # Capture and playback state
        self.input_stream: Optional[pyaudio.Stream] = None
        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        self.output_stream: Optional[pyaudio.Stream] = None

        logger.info("AudioProcessor initialized with 24kHz PCM16 mono audio")

    def start_capture(self) -> None:
        """
        Start capturing audio from the microphone.
        
        Audio is captured in a callback and sent to the VoiceLive connection
        as base64-encoded PCM16 data.
        """
        def _capture_callback(in_data, _frame_count, _time_info, _status_flags):
            """Audio capture callback - runs in PyAudio thread."""
            audio_base64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=audio_base64), 
                self.loop
            )
            return (None, pyaudio.paContinue)

        if self.input_stream:
            return

        self.loop = asyncio.get_event_loop()

        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_capture_callback,
            )
            logger.info("Started audio capture")
        except Exception:
            logger.exception("Failed to start audio capture")
            raise

    def start_playback(self) -> None:
        """
        Initialize the audio playback system.
        
        Audio is played back from a queue in a callback, allowing for
        smooth streaming playback with support for skipping/cancellation.
        """
        if self.output_stream:
            return

        remaining = bytes()

        def _playback_callback(_in_data, frame_count, _time_info, _status_flags):
            """Audio playback callback - runs in PyAudio thread."""
            nonlocal remaining
            frame_count *= pyaudio.get_sample_size(pyaudio.paInt16)

            out = remaining[:frame_count]
            remaining = remaining[frame_count:]

            while len(out) < frame_count:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    out = out + bytes(frame_count - len(out))
                    continue
                except Exception:
                    logger.exception("Error in audio playback")
                    raise

                if not packet or not packet.data:
                    logger.info("End of playback queue.")
                    break

                if packet.seq_num < self.playback_base:
                    if len(remaining) > 0:
                        remaining = bytes()
                    continue

                num_to_take = frame_count - len(out)
                out = out + packet.data[:num_to_take]
                remaining = packet.data[num_to_take:]

            if len(out) >= frame_count:
                return (out, pyaudio.paContinue)
            else:
                return (out, pyaudio.paComplete)

        try:
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_playback_callback
            )
            logger.info("Audio playback system ready")
        except Exception:
            logger.exception("Failed to initialize audio playback")
            raise

    def _get_and_increase_seq_num(self) -> int:
        """Get current sequence number and increment for next use."""
        seq = self.next_seq_num
        self.next_seq_num += 1
        return seq

    def queue_audio(self, audio_data: Optional[bytes]) -> None:
        """
        Queue audio data for playback.
        
        Args:
            audio_data: Raw PCM16 audio bytes to play, or None to signal end
        """
        self.playback_queue.put(
            AudioProcessor.AudioPlaybackPacket(
                seq_num=self._get_and_increase_seq_num(),
                data=audio_data
            )
        )

    def skip_pending_audio(self) -> None:
        """
        Skip all pending audio in the playback queue.
        
        This is useful for implementing barge-in (user interruption)
        where we want to stop the current response immediately.
        """
        self.playback_base = self._get_and_increase_seq_num()

    def shutdown(self) -> None:
        """
        Clean up all audio resources.
        
        This should be called when the voice assistant session ends
        to properly close audio streams and release PyAudio resources.
        """
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        logger.info("Stopped audio capture")

        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        logger.info("Stopped audio playback")

        if self.audio:
            self.audio.terminate()
        logger.info("Audio processor cleaned up")
