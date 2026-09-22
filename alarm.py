import os
import sys
import time
import threading

# Try using pygame for cross-platform MP3 audio playback
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class AlarmManager:
    """Handles non-blocking MP3 audio alerts with cooldown protection."""
    def __init__(self, audio_file="alarm.mp3", cooldown_sec=1.5):
        self.audio_file = audio_file
        self.cooldown_sec = cooldown_sec
        self.last_triggered = 0.0

        # Check if MP3 file exists in root directory
        if not os.path.exists(self.audio_file):
            print(f"[Warning] Audio file '{self.audio_file}' not found in root directory!")

    def _play_sound(self):
        """Plays alarm.mp3 asynchronously without freezing video stream."""
        if HAS_PYGAME and os.path.exists(self.audio_file):
            try:
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"[Audio Error] {e}")
        else:
            # Fallback if pygame is not installed or file is missing
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1200, 400)
            else:
                print("\a", end="", flush=True)

    def trigger(self):
        now = time.time()
        if now - self.last_triggered >= self.cooldown_sec:
            self.last_triggered = now
            # Run sound in a background daemon thread so video playback doesn't lag
            threading.Thread(target=self._play_sound, daemon=True).start()