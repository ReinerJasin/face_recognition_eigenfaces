import cv2
import numpy as np

def preprocess_face(image_path, TARGET_WIDTH, TARGET_HEIGHT):
    """
    Pipeline for the image preprocessing, including these steps:
    1. reading the image from the path as grayscale.
    2. resize to target shape
    3. flatten the image to make it 1 dimensional
    
    Args:
        image_path (string): The path to the input image to be preprocessed
        TARGET_WIDTH (int): The expected width of the preprocessed image result shape (pixel)
        TARGET_HEIGHT (int): The expected height of the preprocessed image result shape in (pixel)
    """
    input_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(input_img, (TARGET_WIDTH, TARGET_HEIGHT))
    image = image.flatten()
    return image

def project_face(image_flattened, mean_faces, eigenfaces):
    face_diff = image_flattened - mean_faces
    projection = face_diff @ eigenfaces
    return projection

def recognize_face(test_projection, train_projections, labels):
    distances = np.linalg.norm(train_projections - test_projection, axis=1)
    min_index = np.argmin(distances)
    return labels[min_index], distances[min_index]

def recognize_face_centroid(test_projection, class_centroids):
    distances = {
        label: np.linalg.norm(test_projection - centroid)
        for label, centroid in class_centroids.items()
    }
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances[predicted_label]

