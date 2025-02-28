#!/usr/bin/env python3

from FaceFilterBase import FaceFilter
import numpy as np
import cv2

class GlassesFilter(FaceFilter):
    """
    A filter that applies glasses to a face based on facial landmarks.
    """
    def __init__(self, assets_path):
        """
        Initializes the GlassesFilter with the path to the assets.
        
        :param assets_path: Path to the assets directory.
        """
        super().__init__(assets_path)
        self.required_landmarks = [33, 263, 1]  # Eye and nose landmarks

    def rotate_image(self, image, angle):
        """
        Rotates an image while maintaining transparency.
        
        :param image: The image to rotate.
        :param angle: The angle to rotate the image.
        :return: The rotated image.
        """
        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        """
        Applies the glasses filter to the given frame based on facial landmarks.
        
        :param frame: The frame to apply the filter to.
        :param face_landmarks: The facial landmarks detected in the frame.
        :param frame_size: The size of the frame.
        :return: The frame with the glasses filter applied.
        """
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Convert to pixel coordinates
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            # Calculate angle of rotation
            dx = x2 - x1
            dy = y2 - y1
            angle = -np.degrees(np.arctan2(dy, dx))
            
            # Calculate size
            eye_distance = np.hypot(dx, dy)
            glasses_width = int(eye_distance * 1.5)
            glasses_img = self.assets[self.current_asset_idx]
            aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            
            # Process image
            resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
            rotated = self.rotate_image(resized, angle)
            
            # Positioning
            center_x = (x1 + x2)//2 - rotated.shape[1]//2
            center_y = (y1 + y2)//2 - rotated.shape[0]//2
            
            # Optimized overlay
            return super().optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

class HatFilter(FaceFilter):
    """
    A filter that applies a hat to a face based on facial landmarks.
    """
    def __init__(self, assets_path):
        """
        Initializes the HatFilter with the path to the assets.
        
        :param assets_path: Path to the assets directory.
        """
        super().__init__(assets_path)
        self.required_landmarks = [103, 332]  # Forehead landmarks

    def rotate_image(self, image, angle):
        """
        Rotates an image while maintaining transparency.
        
        :param image: The image to rotate.
        :param angle: The angle to rotate the image.
        :return: The rotated image.
        """
        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        """
        Applies the hat filter to the given frame based on facial landmarks.
        
        :param frame: The frame to apply the filter to.
        :param face_landmarks: The facial landmarks detected in the frame.
        :param frame_size: The size of the frame.
        :return: The frame with the hat filter applied.
        """
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            forehead_left_side = face_landmarks.landmark[103]
            forehead_right_side = face_landmarks.landmark[332]
            
            # Convert to pixel coordinates
            x1, y1 = int(forehead_left_side.x * w), int(forehead_left_side.y * h)
            x2, y2 = int(forehead_right_side.x * w), int(forehead_right_side.y * h)

            # Calculate angle of rotation
            dx = x2 - x1
            dy = y2 - y1
            angle = -np.degrees(np.arctan2(dy, dx))
            
            # Calculate size
            forehead_distance = np.hypot(dx, dy)
            hat_width = int(forehead_distance * 1.8)
            hat_img = self.assets[self.current_asset_idx]
            aspect_ratio = hat_img.shape[0] / hat_img.shape[1]
            hat_height = int(hat_width * aspect_ratio * 0.8)
            
            # Process image
            resized = cv2.resize(hat_img, (hat_width, hat_height))
            rotated = self.rotate_image(resized, angle)
            
            # Positioning
            center_x = (x1 + x2)//2 - rotated.shape[1]//2
            center_y = (y1 + y2)//2 - rotated.shape[0]//2 - hat_height//3
            
            # Optimized overlay
            return super().optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

class NoseFilter(FaceFilter):
    """
    A filter that applies a nose accessory to a face based on facial landmarks.
    """
    def __init__(self, assets_path):
        """
        Initializes the NoseFilter with the path to the assets.
        
        :param assets_path: Path to the assets directory.
        """
        super().__init__(assets_path)
        self.required_landmarks = [1, 4, 5, 6, 168]  # Nose landmarks

    def rotate_image(self, image, angle):
        """
        Rotates an image with optimized padding.
        
        :param image: The image to rotate.
        :param angle: The angle to rotate the image.
        :return: The rotated image.
        """
        if angle == 0:
            return image

        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular nuevo tamaño con mínima pérdida
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        """
        Applies the nose filter to the given frame based on facial landmarks.
        
        :param frame: The frame to apply the filter to.
        :param face_landmarks: The facial landmarks detected in the frame.
        :param frame_size: The size of the frame.
        :return: The frame with the nose filter applied.
        """
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark
            
            # Main nose landmarks
            nose_tip = landmarks[4]
            nose_bridge = landmarks[6]
            left_nostril = landmarks[98]
            right_nostril = landmarks[327]

            # Convert to pixel coordinates
            x_tip, y_tip = int(nose_tip.x * w), int(nose_tip.y * h)
            x_bridge, y_bridge = int(nose_bridge.x * w), int(nose_bridge.y * h)
            x_left, y_left = int(left_nostril.x * w), int(left_nostril.y * h)
            x_right, y_right = int(right_nostril.x * w), int(right_nostril.y * h)

            if not all([self._is_visible(lm, w, h) for lm in [nose_tip, nose_bridge]]):
                return frame

            # Calculate size and position of the nose
            nose_width = abs(x_right - x_left)
            vertical_distance = abs(y_bridge - y_tip)
            nose_height = int(vertical_distance * 1.2)
            
            dx = x_right - x_left
            dy = y_right - y_left
            angle = -np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0

            nose_asset = self.assets[self.current_asset_idx]
            
            target_width = int(nose_width * 1.5)
            aspect_ratio = nose_asset.shape[0] / nose_asset.shape[1]
            target_height = int(target_width * aspect_ratio)
            
            # Process image
            resized = cv2.resize(nose_asset, (target_width, target_height))
            rotated = self.rotate_image(resized, angle)

            # Positioning
            pos_x = x_tip - rotated.shape[1] // 2
            pos_y = y_tip - rotated.shape[0] // 2

            if not self._is_valid_position(pos_x, pos_y, rotated.shape, w, h):
                return frame

            # Optimized overlay
            return super().optimized_overlay(frame, rotated, pos_x, pos_y)

        except (IndexError, cv2.error) as e:
            print(f"Error aplicando filtro de nariz: {e}")
            return frame

    def _is_visible(self, landmark, img_w, img_h):
        """
        Checks if a landmark is within the visible area.
        
        :param landmark: The landmark to check.
        :param img_w: The width of the image.
        :param img_h: The height of the image.
        :return: True if the landmark is visible, False otherwise.
        """
        return (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1 and
                img_w * 0.1 <= landmark.x * img_w <= img_w * 0.9 and
                img_h * 0.1 <= landmark.y * img_h <= img_h * 0.9)

    def _is_valid_position(self, x, y, shape, img_w, img_h):
        """
        Validates that the position is within the image boundaries.
        
        :param x: The x-coordinate of the position.
        :param y: The y-coordinate of the position.
        :param shape: The shape of the asset.
        :param img_w: The width of the image.
        :param img_h: The height of the image.
        :return: True if the position is valid, False otherwise.
        """
        h_asset, w_asset = shape[:2]
        return (x >= -w_asset * 0.2 and 
                y >= -h_asset * 0.2 and
                x + w_asset <= img_w * 1.2 and
                y + h_asset <= img_h * 1.2)

class MouthFilter(FaceFilter):
    """
    A filter that applies a mouth accessory to a face based on facial landmarks.
    """
    def __init__(self, assets_path):
        """
        Initializes the MouthFilter with the path to the assets.
        
        :param assets_path: Path to the assets directory.
        """
        super().__init__(assets_path)
        self.required_landmarks = [
            61, 291,  # Mouth corners
            13, 14,   # Lip midline
            78, 308,  # Upper lip contour
            37, 267,  # Lower lip contour
        ]

    def rotate_image(self, image, angle):
        """
        Rotates an image while maintaining transparency.
        
        :param image: The image to rotate.
        :param angle: The angle to rotate the image.
        :return: The rotated image.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0, 2] += (new_w - w) // 2
        rot_mat[1, 2] += (new_h - h) // 2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        """
        Applies the mouth filter to the given frame based on facial landmarks.
        
        :param frame: The frame to apply the filter to.
        :param face_landmarks: The facial landmarks detected in the frame.
        :param frame_size: The size of the frame.
        :return: The frame with the mouth filter applied.
        """
        if not self.assets or not face_landmarks:
            return frame

        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark

            # Main mouth landmarks
            left_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
            right_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
            upper_lip = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            lower_lip = (int(landmarks[17].x * w), int(landmarks[17].y * h))

            # Calculate size
            mouth_width = abs(right_corner[0] - left_corner[0])
            mouth_height = abs(lower_lip[1] - upper_lip[1])

            if mouth_width == 0 or mouth_height == 0:
                return frame

            mouth_asset = self.assets[self.current_asset_idx]

            aspect_ratio = mouth_asset.shape[0] / mouth_asset.shape[1]
            target_width = int(mouth_width)
            target_height = int(target_width * aspect_ratio)
            resized_asset = cv2.resize(mouth_asset, (target_width, target_height))

            dx = right_corner[0] - left_corner[0]
            dy = right_corner[1] - left_corner[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            rotated_asset = self.rotate_image(resized_asset, angle)

            # Positioning
            center_x = (left_corner[0] + right_corner[0]) // 2 - rotated_asset.shape[1] // 2
            center_y = (upper_lip[1] + lower_lip[1]) // 2 - rotated_asset.shape[0] // 2

            if not self._validate_position(center_x, center_y, rotated_asset.shape, w, h):
                return frame

            # Optimized overlay
            return super().optimized_overlay(frame, rotated_asset, center_x, center_y)
            
        except (IndexError, cv2.error) as e:
            print(f"Error en filtro bucal: {str(e)}")
            return frame

    def _validate_position(self, x, y, shape, img_w, img_h):
        """
        Validates that the filter is within reasonable facial boundaries.
        
        :param x: The x-coordinate of the position.
        :param y: The y-coordinate of the position.
        :param shape: The shape of the asset.
        :param img_w: The width of the image.
        :param img_h: The height of the image.
        :return: True if the position is valid, False otherwise.
        """
        h_asset, w_asset = shape[:2]
        return (
            x > -w_asset * 0.2 and
            y > -h_asset * 0.2 and
            x + w_asset < img_w * 1.2 and
            y + h_asset < img_h * 1.2 and
            w_asset > 10 and  # Tamaño mínimo
            h_asset > 10
        )

class FaceMaskFilter(FaceFilter):
    """
    A filter that applies a face mask to a face based on facial landmarks.
    """
    def __init__(self, assets_path):
        """
        Initializes the FaceMaskFilter with the path to the assets.
        
        :param assets_path: Path to the assets directory.
        """
        super().__init__(assets_path)
        self.required_landmarks = [
            10,  # Top of the forehead
            152,  # Bottom of the chin
            234,  # Left side of the face
            454,  # Right side of the face
            61, 291,  # Mouth corners
            1,  # Nose
            5,  # Nose tip
            2,  # Right eyebrow
            5,  # Left eyebrow
        ]

    def apply_filter(self, frame, face_landmarks, frame_size):
        """
        Applies the face mask filter to the given frame based on facial landmarks.
        
        :param frame: The frame to apply the filter to.
        :param face_landmarks: The facial landmarks detected in the frame.
        :param frame_size: The size of the frame.
        :return: The frame with the face mask filter applied.
        """
        if not self.assets or not face_landmarks:
            return frame

        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark

            # Main face landmarks
            top_forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
            chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            left_side = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            right_side = (int(landmarks[454].x * w), int(landmarks[454].y * h))

            # Calculate size
            face_width = abs(right_side[0] - left_side[0])
            face_height = abs(chin[1] - top_forehead[1])

            if face_width == 0 or face_height == 0:
                return frame

            mask_asset = self.assets[self.current_asset_idx]

            aspect_ratio = mask_asset.shape[0] / mask_asset.shape[1]
            target_width = int(face_width * 2.0)
            target_height = int(target_width * aspect_ratio)
            resized_asset = cv2.resize(mask_asset, (target_width, target_height))

            dx = right_side[0] - left_side[0]
            dy = right_side[1] - left_side[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            rotated_asset = self.rotate_image(resized_asset, angle)

            # Positioning
            center_x = (left_side[0] + right_side[0]) // 2 - rotated_asset.shape[1] // 2
            center_y = (top_forehead[1] + chin[1]) // 2 - rotated_asset.shape[0] // 2

            if not self._validate_position(center_x, center_y, rotated_asset.shape, w, h):
                return frame

            # Optimized overlay
            return super().optimized_overlay(frame, rotated_asset, center_x, center_y)

        except (IndexError, cv2.error) as e:
            print(f"Error en filtro de máscara facial: {str(e)}")
            return frame

    def rotate_image(self, image, angle):
        """
        Rotates an image while maintaining transparency.
        
        :param image: The image to rotate.
        :param angle: The angle to rotate the image.
        :return: The rotated image.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0, 2] += (new_w - w) // 2
        rot_mat[1, 2] += (new_h - h) // 2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def _validate_position(self, x, y, shape, img_w, img_h):
        """
        Validates that the filter is within reasonable facial boundaries.
        
        :param x: The x-coordinate of the position.
        :param y: The y-coordinate of the position.
        :param shape: The shape of the asset.
        :param img_w: The width of the image.
        :param img_h: The height of the image.
        :return: True if the position is valid, False otherwise.
        """
        h_asset, w_asset = shape[:2]
        return (
            x > -w_asset * 0.2 and
            y > -h_asset * 0.2 and
            x + w_asset < img_w * 1.2 and
            y + h_asset < img_h * 1.2 and
            w_asset > 10 and  # Tamaño mínimo
            h_asset > 10
        )
