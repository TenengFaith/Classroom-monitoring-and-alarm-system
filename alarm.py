import os
import time
import threading

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class Alarm:
    """
    Asynchronous sound alarm manager with a cooldown period.
    """
    def __init__(self, sound_path="alarm.mp3", cooldown_sec=2.0):
        self.sound_path = sound_path
        self.cooldown_sec = cooldown_sec
        self.last_played_time = 0.0

        if PYGAME_AVAILABLE and os.path.exists(self.sound_path):
            pygame.mixer.init()
            self.sound = pygame.mixer.Sound(self.sound_path)
            self.audio_ready = True
            print(f"[ALARM] Sound engine ready with file: '{self.sound_path}'")
        else:
            self.audio_ready = False
            if not PYGAME_AVAILABLE:
                print("[WARNING] 'pygame' not installed. Running alarm in terminal fallback mode.")
            elif not os.path.exists(self.sound_path):
                print(f"[WARNING] Sound file '{self.sound_path}' not found in root directory. Running in terminal mode.")

    def trigger(self):
        """Triggers alarm sound if cooldown period has passed."""
        now = time.time()
        if now - self.last_played_time >= self.cooldown_sec:
            self.last_played_time = now
            threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self):
        if self.audio_ready:
            try:
                self.sound.play()
            except Exception as e:
                print(f"[ALARM ERROR] {e}")
        else:
            print("\a[ALARM BEEP] Suspicious student behavior detected!")