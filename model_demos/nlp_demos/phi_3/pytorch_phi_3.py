import pybuda

available_devices = pybuda.detect_available_devices()
print(f"Available devices: {available_devices}")
