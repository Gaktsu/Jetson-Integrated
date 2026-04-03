from pipeline.shared_state import SharedState
from pipeline.capture import capture_loop, start_capture_threads
from pipeline.inference import inference_loop, start_inference_thread
from pipeline.recorder import save_loop, start_save_thread
from pipeline.sensors import start_sensor_threads
