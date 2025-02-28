#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mediapipe as mp
import time
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from Filters import GlassesFilter, HatFilter, NoseFilter, MouthFilter, FaceMaskFilter

class FaceFilterNode(Node):
    """
    Nodo ROS2 para aplicar filtros faciales usando Mediapipe y OpenCV, suscrito al topic image_raw.
    """
    def __init__(self):
        super().__init__('face_filter_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        
        # Inicialización de Mediapipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.package_share_path = get_package_share_directory('buddy-filters')
        self.assets_path = os.path.join(self.package_share_path, 'imgs')
        
        # Filtros activos
        self.active_filters = {
            'glasses': GlassesFilter(self.assets_path + "/glasses/"),
            'hat': HatFilter(self.assets_path + "/hats/"),
            'nose': NoseFilter(self.assets_path + "/noses/"),
            'mouth': MouthFilter(self.assets_path + "/mouths/"),
        }
        self.is_face_selected = False
        self.prev_time = time.time()
        self.frame_count = 0

        self.draw = mp.solutions.drawing_utils

    def flip_frame(self, frame):
        """Voltea el frame horizontalmente."""
        return cv2.flip(frame, 1)

    def image_callback(self, msg):
        """Callback para procesar cada frame recibido."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error al convertir imagen: {str(e)}')
            return

        frame = self.flip_frame(cv_image)
        self.frame_count += 1

        # Procesamiento con Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for filter_name, filter_obj in self.active_filters.items():
                    frame = filter_obj.apply_filter(frame, face_landmarks, frame.shape[:2])

        # Cálculo de FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        #self.draw.draw_landmarks(frame, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_TESSELATION)
        # Mostrar frame
        cv2.imshow('Face Filters', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Manejo de teclado
        if key == ord('e') and not self.is_face_selected:
            self.active_filters['hat'].current_asset_idx = (self.active_filters['hat'].current_asset_idx + 1) % len(self.active_filters['hat'].assets)
        elif key == ord('q') and not self.is_face_selected:
            self.active_filters['hat'].current_asset_idx = (self.active_filters['hat'].current_asset_idx - 1) % len(self.active_filters['hat'].assets)
        elif key == ord('d') and not self.is_face_selected:
            self.active_filters['glasses'].current_asset_idx = (self.active_filters['glasses'].current_asset_idx + 1) % len(self.active_filters['glasses'].assets)
        elif key == ord('a') and not self.is_face_selected:
            self.active_filters['glasses'].current_asset_idx = (self.active_filters['glasses'].current_asset_idx - 1) % len(self.active_filters['glasses'].assets)
        elif key == ord('w') and not self.is_face_selected:
            self.active_filters['nose'].current_asset_idx = (self.active_filters['nose'].current_asset_idx + 1) % len(self.active_filters['nose'].assets)
        elif key == ord('s') and not self.is_face_selected:
            self.active_filters['nose'].current_asset_idx = (self.active_filters['nose'].current_asset_idx - 1) % len(self.active_filters['nose'].assets)
        elif key == ord('c') and not self.is_face_selected:
            self.active_filters['mouth'].current_asset_idx = (self.active_filters['mouth'].current_asset_idx + 1) % len(self.active_filters['mouth'].assets)
        elif key == ord('z') and not self.is_face_selected:
            self.active_filters['mouth'].current_asset_idx = (self.active_filters['mouth'].current_asset_idx - 1) % len(self.active_filters['mouth'].assets)
        elif key == ord('f'):
            self.is_face_selected = not self.is_face_selected
            if self.is_face_selected:
                self.active_filters = {'face': FaceMaskFilter(self.assets_path + "/faces/")}
            else:
                self.active_filters = {
                    'glasses': GlassesFilter(self.assets_path + "/glasses/"),
                'hat': HatFilter(self.assets_path + "/hats/"),
                'nose': NoseFilter(self.assets_path + "/noses/"),
                'mouth': MouthFilter(self.assets_path + "/mouths/"),
                }
        elif key == ord('m') and self.is_face_selected:
            self.active_filters['face'].current_asset_idx = (self.active_filters['face'].current_asset_idx + 1) % len(self.active_filters['face'].assets)
        elif key == ord('n') and self.is_face_selected:
            self.active_filters['face'].current_asset_idx = (self.active_filters['face'].current_asset_idx - 1) % len(self.active_filters['face'].assets)
        elif key == ord('l'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = FaceFilterNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f'Error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()