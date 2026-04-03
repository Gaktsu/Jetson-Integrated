"""
부저 제어 (능동 부저: HIGH=ON, LOW=OFF)
"""

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


class Buzzer:
    """부저 제어 클래스 (능동 부저)"""

    def __init__(self, pin: int = 32, use_board: bool = True):
        self.pin = pin
        self.use_board = use_board
        self.is_active = False
        self._initialized = False

    def start(self) -> bool:
        """부저 초기화"""
        if GPIO is None:
            print("RPi.GPIO를 사용할 수 없습니다. 부저는 로그만 출력됩니다.")
            self._initialized = False
            return False

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD if self.use_board else GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self.is_active = False
        self._initialized = True
        return True

    def activate(self):
        """부저 켜기 (HIGH)"""
        if GPIO is None or not self._initialized:
            if not self.is_active:
                print("부저 활성화")
            self.is_active = True
            return

        GPIO.output(self.pin, GPIO.HIGH)
        self.is_active = True

    def deactivate(self):
        """부저 끄기 (LOW)"""
        if GPIO is None or not self._initialized:
            if self.is_active:
                print("부저 비활성화")
            self.is_active = False
            return

        GPIO.output(self.pin, GPIO.LOW)
        self.is_active = False

    def stop(self):
        """부저 정리"""
        self.deactivate()
        if GPIO is not None and self._initialized:
            GPIO.cleanup(self.pin)
            self._initialized = False
