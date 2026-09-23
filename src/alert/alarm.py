import os
import time
import threading

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ALARM_SOUND = os.path.join(BASE_DIR, "assets", "alarm.mp3")


class Alarm:
    """
    Asynchronous sound alarm manager with cooldown periods.
    Uses pygame.mixer if available, fallback to system beep.
    """
    def __init__(self, sound_path=None, cooldown_sec=2.0):
        self.sound_path = sound_path or DEFAULT_ALARM_SOUND
        self.cooldown_sec = cooldown_sec
        self.last_played_time = 0.0

        if PYGAME_AVAILABLE and os.path.exists(self.sound_path):
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(self.sound_path)
                self.audio_ready = True
                print(f"[ALARM] Sound engine ready with file: '{self.sound_path}'")
            except Exception as e:
                self.audio_ready = False
                print(f"[WARNING] Could not initialize sound file: {e}")
        else:
            self.audio_ready = False
            if not PYGAME_AVAILABLE:
                print("[WARNING] 'pygame' not installed. Running alarm in fallback mode.")
            elif not os.path.exists(self.sound_path):
                print(f"[WARNING] Sound file '{self.sound_path}' not found. Running in fallback mode.")

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
            print("\a[ALARM BEEP] Suspicious behavior detected!")
