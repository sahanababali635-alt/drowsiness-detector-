from simple_detector import CameraSensor

sensor = CameraSensor(camera_index=0)
if sensor.is_ready():
    while True:
        ret, frame = sensor.read()
        if ret:
            data = sensor.get_drowsiness_sensor_data()
            print(f"Drowsy: {data['is_drowsy']}, EAR: {data['eye_aspect_ratio']:.2f}")
            # Use sensor data for your application