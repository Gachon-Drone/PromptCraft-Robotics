import airsim
import math
import numpy as np
import logging
import os 
from datetime import datetime
import json
import time
from ultralytics import YOLO
from PIL import Image
import glob

last_command_time = time.time()

OBJECT_LOCATIONS_FILE = 'object_locations.json'

# Configure logging
logging.basicConfig(filename='airsim_wrapper.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# YOLO World model for open-vocabulary detection
YOLO_WORLD_MODEL = None


class AirSimWrapper:

    def __init__(self, ground_level_offset=20.0):
        """
        Initialize AirSim wrapper.
        
        Args:
            ground_level_offset: Distance from origin (Z=0) to ground in meters.
                               Default is 20.0 (ground is at Z=20 in NED coordinates)
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        logging.info('Connected to AirSim')
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logging.info('Drone is armed and API control is enabled')

        initial_wind = airsim.Vector3r(0, 0, 0)
        self.client.simSetWind(initial_wind)

        # Ground level configuration
        self.ground_level = ground_level_offset  # Ground is at Z = ground_level_offset
        self.min_safe_altitude = 1.0  # Minimum 1 meter above ground
        print(f"Ground level set at Z={self.ground_level} meters")
        print(f"Safe flying range: Z=0 to Z={self.ground_level - self.min_safe_altitude}")

        # Initialize variables for logging
        self.logging_enabled = False
        self.log_duration = None
        self.log_start_time = None
        self.log_file = None
        self.log_data = []
        
        # Initialize YOLO models
        self.yolo_model = None
        self.yolo_world_model = None
        
        global original_position
        original_position = self.load_original_position()
        if original_position is None:
            original_position = self.get_current_position()
            self.save_original_position(original_position)

    def set_wind(self, x, y, z):
        wind = airsim.Vector3r(x, y, z)
        self.client.simSetWind(wind)
        print(f"Wind set to: X={x} m/s, Y={y} m/s, Z={z} m/s")

    ORIGINAL_POSITION_FILE = 'original_position.json'
    wind = airsim.Vector3r(5, 0, 0)
    
    def z(self):
        return self.client
        
    def takeoff(self):
        try:
            self.client.takeoffAsync().join()
            print("Takeoff successful.")
        except Exception as e:
            print(f"Error during takeoff: {str(e)}")

    def save_original_position(self, position):
        """Save the original position to a file."""
        with open(self.ORIGINAL_POSITION_FILE, 'w') as f:
            json.dump(position, f)

    def load_original_position(self):
        """Load the original position from a file."""
        if os.path.exists(self.ORIGINAL_POSITION_FILE):
            with open(self.ORIGINAL_POSITION_FILE, 'r') as f:
                return json.load(f)
        return None
    
    def land_at_original_position(self):
        print("Returning to original position...")
        print(f"Original position: {original_position}")
        print(f"Current position: {self.get_current_position()}")
        self.client.moveToPositionAsync(
            int(original_position['x_val']), 
            int(original_position['y_val']), 
            int(original_position['z_val']), 
            2
        ).join()
        time.sleep(2)
        self.client.landAsync().join()
        return True
    

    def search_for_object_and_move_to_it(self, obj_name, use_world_model=False):
        """
        Search for a specific object and move towards it.
        
        Args:
            obj_name: Name of the object to search for
            use_world_model: If True, use YOLO-World to detect custom objects
        """
        global last_command_time
        
        print(f"Searching for {obj_name}...")
       
        start_time1 = time.time()
        self.adjust_camera_pitch(-15)

        object_found = False
        rot = 0
        rotation_attempts = 0
        max_rotations = 8  

        while rotation_attempts < max_rotations:
            img = self.get_image()
            
            # Use YOLO-World if requested, with the specific object as custom class
            if use_world_model:
                obj_list, obj_locs = self.detect_objects(img, use_world_model=True, custom_classes=[obj_name])
            else:
                obj_list, obj_locs = self.detect_objects(img)

            obj_name_normalized = obj_name.lower()
            detected_objects = [obj.lower() for obj in obj_list]
            print(f"Detected objects (normalized): {detected_objects}")

            if obj_name_normalized in detected_objects:
                obj_idx = detected_objects.index(obj_name_normalized)
                print(f"Object '{obj_name}' found!")
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
                print(f"Command execution to find the object time: {elapsed_time1:.2f} seconds")

                distance, angle, height = self.get_object_distance_and_angle_and_height(img, obj_locs[obj_idx])

                if distance is not None and distance > 15:
                    print(f"Distance to {obj_name}: {distance:.2f} cm. Moving closer...")
                    self.move_towards_object(distance, angle, height, obj_name, rot)
                    object_found = True
                    break  

                elif distance is not None and distance <= 15:
                    print(f"Reached {obj_name}. Stopping search.")
                    return

                else:
                    print(f"Object '{obj_name}' detected, but could not determine distance. Trying again...")

            if not object_found:
                print(f"{obj_name} not found. Rotating to search...")
                self.turn_left()
                rot = rot + 1
                rotation_attempts += 1
                if (rot > 3):
                    rot = 0

    def turn_left(self):
        self.client.rotateByYawRateAsync(-45, 1).join()
        
    def turn_right(self):
        self.client.rotateByYawRateAsync(45, 1).join()

    def move_towards_object(self, distance, angle, height, obj_name, rot):
        step_distance = 0.5  
        step_height = 0.05  
        proximity_threshold = 30  

        print(f"Moving towards {obj_name}...")
        distance = distance/100

        current_pos = self.get_current_position()

        if rot == 0:
            dx = int(current_pos['x_val'])
            dy = int(current_pos['y_val'])-distance
            dz = int(current_pos['z_val'])
        elif rot == 1:
            dx = int(current_pos['x_val'])-distance
            dy = int(current_pos['y_val'])
            dz = int(current_pos['z_val'])     
        elif rot == 2:
            dx = int(current_pos['x_val'])
            dy = int(current_pos['y_val'])+distance
            dz = int(current_pos['z_val'])           
        else:
            dx = int(current_pos['x_val'])+distance
            dy = int(current_pos['y_val'])
            dz = int(current_pos['z_val'])
        self.fly_to([dx,dy,dz])

        print(f"Reached {obj_name}!")
        
        result_dirs = glob.glob('results/output*')
        if result_dirs:
            latest_result_dir = max(result_dirs, key=os.path.getctime)
            
            image_files = glob.glob(os.path.join(latest_result_dir, '*.jpg')) + glob.glob(os.path.join(latest_result_dir, '*.png'))
            if image_files:
                latest_image_file = image_files[0]
                img = Image.open(latest_image_file)
                img.show()
                print(f"Displaying image: {latest_image_file}")
            else:
                print("No images found in the latest results directory.")
                return
        else:
            print("No results directories found.")
            return  

        user_input = input("Does the image look okay? (yes/no): ").lower()

        if user_input == 'yes':
            print("Proceeding with the current image.")
                    
            current_position = self.get_current_position()
            x = current_position['x_val']
            y = current_position['y_val']
            z = current_position['z_val']

            command_text = f'**Command**: "Go to {obj_name}"\n   Response:\n  ```python\n  aw.takeoff()  # Command to take off\n  aw.fly_to([{x}, {y}, {z}])  # Command to fly to {obj_name}\n  ```\n'
                    
            airsim_basics_file = 'prompts/airsim_basic.txt'

            try:
                with open(airsim_basics_file, 'a') as file:
                    file.write('\n' + command_text)
                print(f"Appended new command to {airsim_basics_file}")
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")
                
        else:
            print("The image does not look okay. Retrying or taking necessary action.")    

    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        ])
    
        if len(responses) > 0:
            response = responses[0]
        
            if response.image_data_float:
                depth_image = np.array(response.image_data_float, dtype=np.float32)
                depth_image = depth_image.reshape(response.height, response.width)
                return depth_image
            else:
                raise ValueError("Depth image data is empty or invalid.")
        else:
            raise ValueError("No image response received from AirSim.")

    def get_object_distance_and_angle_and_height(self, img, bbox):
        img_width, img_height = img.size

        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        depth_image = self.get_depth_image()

        if depth_image is None or center_x < 0 or center_x >= depth_image.shape[1] or center_y < 0 or center_y >= depth_image.shape[0]:
            logging.error("Invalid depth image or bounding box center out of bounds.")
            return None, None, None

        depth = depth_image[center_y, center_x]

        if depth <= 0 or depth > 1000:
            logging.error(f"Invalid depth value: {depth}.")
            return None, None, None

        distance = depth * 100

        angle = math.atan2(center_x - img_width / 2, img_height / 2 - center_y)

        height = (bbox[3] - bbox[1]) / img_height
    
        print(angle, depth)
        return distance, angle, height

    def adjust_camera_pitch(self, pitch_angle):
        current_orientation = self.client.simGetVehiclePose().orientation
        roll, pitch, yaw = airsim.to_eularian_angles(current_orientation)

        new_pitch = math.radians(pitch_angle)

        new_orientation = airsim.to_quaternion(new_pitch, roll, yaw)
        self.client.simSetVehiclePose(airsim.Pose(self.client.simGetVehiclePose().position, new_orientation), True)

        print(f"Adjusted drone pitch to {pitch_angle} degrees.")

    def detect_objects(self, img, use_world_model=False, custom_classes=None):
        """
        Detect objects in an image.
        
        Args:
            img: PIL Image object
            use_world_model: If True, use YOLO-World for custom object detection
            custom_classes: List of object names to search for (e.g., ["bottle", "laptop"])
        """
        obj_list, obj_locs = self.process_image_with_yolo(img, use_world_model, custom_classes)
        if not obj_list:
            print("No objects detected in the current image.")
        else:
            print(f"Detected objects: {obj_list}")
        return obj_list, obj_locs

    def get_image(self):
        return self.capture_image()  
    
    def get_current_position(self):
        """
        Get the current position of the drone.
        
        Returns:
            dict: Position with 'x_val', 'y_val', 'z_val', and 'altitude_agl' (above ground level)
        """
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        altitude_agl = self.ground_level - position.z_val
        
        return {
            "x_val": position.x_val,
            "y_val": position.y_val,
            "z_val": position.z_val,
            "altitude_agl": altitude_agl  # Altitude Above Ground Level
        }

    def capture_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if not responses or len(responses) == 0:
            logging.error("No image received from AirSim")
            return None

        for idx, response in enumerate(responses):
            if response.width == 0 or response.height == 0:
                logging.error("Invalid image data received from AirSim")
                return None

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            if img1d.size != response.width * response.height * 3:
                logging.error("Mismatch in image data size")
                return None

            img_rgb = img1d.reshape(response.height, response.width, 3)
        
            if not os.path.exists('captures'):
                os.makedirs('captures')
        
            filename = f"capture_{idx}.png"
            file_path = os.path.join('captures', filename)
        
            airsim.write_png(file_path, img_rgb)
            logging.info(f"Image captured and saved as {filename}")

            return Image.fromarray(img_rgb)
    
        logging.error("Failed to capture image")
        return None

    def process_image_with_yolo(self, image_path, use_world_model=False, custom_classes=None):
        """
        Process image with YOLO model.
        
        Args:
            image_path: Path to image or PIL Image object
            use_world_model: If True, use YOLO-World for open-vocabulary detection
            custom_classes: List of custom class names for YOLO-World (e.g., ["person", "car", "dog"])
        
        Returns:
            detected_objects: List of detected object names
            bounding_boxes: List of bounding box coordinates
        """
        if use_world_model:
            # Use YOLO-World for open-vocabulary detection
            if self.yolo_world_model is None:
                print("Loading YOLO-World model...")
                self.yolo_world_model = YOLO("yolov8x-worldv2.pt")
                logging.info("YOLO-World model loaded")
            
            model = self.yolo_world_model
            
            # Set custom classes if provided
            if custom_classes:
                model.set_classes(custom_classes)
                print(f"YOLO-World searching for: {custom_classes}")
        else:
            # Use standard YOLO model
            if self.yolo_model is None:
                print("Loading standard YOLO model...")
                self.yolo_model = YOLO("yolov8x.pt")
                logging.info("Standard YOLO model loaded")
            
            model = self.yolo_model
        
        results = model.predict(image_path, save=True, imgsz=640, conf=0.5, device='0', 
                               project="results", name="output", show_boxes=True)
        
        detected_objects = []
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                detected_objects.append(result.names[int(box.cls[0])])
                bbox = box.xyxy[0].cpu().numpy()
                bounding_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])  
        
        logging.info(f"YOLO processed the image: {detected_objects}")
        return detected_objects, bounding_boxes

    def capture_and_process_image(self, use_world_model=False, custom_classes=None):
        """
        Capture and process image with YOLO.
        
        Args:
            use_world_model: If True, use YOLO-World
            custom_classes: List of custom object names to detect
        """
        image_path = self.capture_image()
        detected_objects, bounding_boxes = self.process_image_with_yolo(
            image_path, use_world_model, custom_classes
        )
        return detected_objects, bounding_boxes        
    
    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def check_altitude_safety(self, z_coordinate):
        """
        Check if the Z coordinate is safe (not too close to ground or below it).
        
        Args:
            z_coordinate: Z coordinate to check
            
        Returns:
            safe_z: Safe Z coordinate (clamped if necessary)
            is_safe: Boolean indicating if original coordinate was safe
        """
        max_safe_z = self.ground_level - self.min_safe_altitude
        
        if z_coordinate > max_safe_z:
            print(f"WARNING: Z={z_coordinate} is too close to ground (ground at Z={self.ground_level})")
            print(f"Clamping to safe altitude: Z={max_safe_z}")
            return max_safe_z, False
        
        if z_coordinate < 0:
            # Negative Z is fine (going up), but warn if too high
            if abs(z_coordinate) > 100:  # Arbitrary high altitude warning
                print(f"WARNING: Flying very high at Z={z_coordinate} meters")
        
        return z_coordinate, True
    
    def get_altitude_above_ground(self):
        """
        Get current altitude above ground level.
        
        Returns:
            altitude: Height above ground in meters (positive value)
        """
        current_pos = self.get_current_position()
        altitude = self.ground_level - current_pos['z_val']
        return altitude
    
    def fly_to(self, point, check_safety=True):
        """
        Fly to specified position with safety checks.
        
        Args:
            point: [x, y, z] coordinates
            check_safety: If True, check altitude safety before flying
        """
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        
        # Safety check
        if check_safety:
            safe_z, is_safe = self.check_altitude_safety(z)
            if not is_safe:
                z = safe_z
                print(f"Flying to safe position: [{x}, {y}, {z}]")
        
        if z > self.ground_level - self.min_safe_altitude:
            self.client.moveToPositionAsync(x, y, self.ground_level - self.min_safe_altitude, 5).join()
        else:
            self.client.moveToPositionAsync(x, y, z, 5).join()
        
        # Print altitude after movement
        altitude = self.get_altitude_above_ground()
        print(f"Current altitude above ground: {altitude:.2f} meters")
    
    def fly_to_object(self, object_name):
        """Move the drone to the specified object using its coordinates."""
        try:
            with open(OBJECT_LOCATIONS_FILE, 'r') as f:
                locations = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return
        
        if object_name in locations:
            location = locations[object_name]
            self.client.moveToPositionAsync(
                location['x'],
                location['y'],
                location['z'],
                5
            ).join()
            print(f"Moved to {object_name} at {location}")
        else:
            print(f"Object {object_name} not found in locations file.")
                    
    def fly_path(self, points, check_safety=True):
        """
        Fly through multiple waypoints with safety checks.
        
        Args:
            points: List of [x, y, z] coordinates
            check_safety: If True, check altitude safety for each point
        """
        airsim_points = []
        for point in points:
            x, y, z = point[0], point[1], point[2]
            
            # Safety check
            if check_safety:
                z, _ = self.check_altitude_safety(z)
            
            if z > 0:
                airsim_points.append(airsim.Vector3r(x, y, -z))
            else:
                airsim_points.append(airsim.Vector3r(x, y, z))
        
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, 
                                    airsim.YawMode(False, 0), 20, 1).join()
        
        # Print final altitude
        altitude = self.get_altitude_above_ground()
        print(f"Path complete. Current altitude above ground: {altitude:.2f} meters")

    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    def set_weather(self, weather):
        if weather == "snowy_day":
            self.client.simEnableWeather(True)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
            logging.info('Weather set to snowy day')
        else:
            logging.warning(f'Invalid weather condition: {weather}')

    def get_sensor_snapshot(self):
        """Print comprehensive sensor data including altitude above ground."""
        state = self.client.getMultirotorState()

        orientation_q = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.utils.to_eularian_angles(orientation_q)

        position = state.kinematics_estimated.position
        x, y, z = position.x_val, position.y_val, position.z_val
        
        altitude_agl = self.ground_level - z

        gps_data = self.client.getGpsData()
        lat, lon, alt = gps_data.gnss.geo_point.latitude, gps_data.gnss.geo_point.longitude, gps_data.gnss.geo_point.altitude

        print(f"Orientation (Roll, Pitch, Yaw): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
        print(f"Position (X, Y, Z): {x:.2f}, {y:.2f}, {z:.2f}")
        print(f"Altitude above ground: {altitude_agl:.2f} meters")
        print(f"GPS (Latitude, Longitude, Altitude): {lat}, {lon}, {alt}")