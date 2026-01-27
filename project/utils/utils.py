import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_face(image_path, TARGET_WIDTH, TARGET_HEIGHT, display_plot=False):
    """
    Pipeline for the image preprocessing, including these steps:
    1. reading the image from the path as grayscale.
    2. resize to target shape
    3. flatten the image to make it 1 dimensional
    
    Args:
        image_path (string): The path to the input image to be preprocessed
        TARGET_WIDTH (int): The expected width of the preprocessed image result shape (pixel)
        TARGET_HEIGHT (int): The expected height of the preprocessed image result shape in (pixel)
        display_plot (bool): trigger to display output plot
    """
    input_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(input_img, (TARGET_WIDTH, TARGET_HEIGHT))
    
    if display_plot:
        plt.imshow(image, cmap='gray')
        plt.show()
    
    image = image.flatten()
    
    return image

def project_face(image_flattened, mean_faces, eigenfaces):
    face_diff = image_flattened - mean_faces
    projection = face_diff @ eigenfaces
    return projection

def euclidean_distance(vec1, vec2, axis=None):
    return np.linalg.norm(vec1 - vec2, axis=axis)


def compute_min_distance(test_projection, train_projections, labels):
    distances = euclidean_distance(train_projections, test_projection, axis=1)
    print(f'distances: {distances}')
    
    min_index = np.argmin(distances)
    print(f'min_index: {min_index}')
    
    return labels[min_index], distances[min_index]

def predict_face(
    image_path,
    mean_faces,
    eigenfaces,
    train_projections,
    label_list,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    threshold=None,
    display_plot=False
    ):

    test_face = preprocess_face(image_path, TARGET_WIDTH=TARGET_WIDTH, TARGET_HEIGHT=TARGET_HEIGHT, display_plot=display_plot)
    test_projection = project_face(test_face, mean_faces, eigenfaces)

    predicted_label, distance = compute_min_distance(
        test_projection,
        train_projections,
        label_list
    )

    print("\nW/O threshold:")
    print("Predicted label:", predicted_label)
    print("Distance:", distance)

    print(f"\nWith threshold: {threshold}")
    if threshold is not None and distance > threshold:
        print("Unknown face")
    else:
        print("Recognized as:", predicted_label)





def recognize_face_centroid(test_projection, class_centroids):
    
    
    distances = {
        label: euclidean_distance(test_projection, centroid)
        for label, centroid in class_centroids.items()
    }
    
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances[predicted_label]

def compute_distances_to_centroids(test_projection, class_centroids):
    distances = {}
    
    for label, centroid in class_centroids.items():
        distances[label] = euclidean_distance(test_projection, centroid)
    return distances

def predict_label_from_distances(distances, label_mapping=None, threshold=None):
    predicted_label = min(distances, key=distances.get)
    min_distance = distances[predicted_label]

    if threshold is not None and min_distance > threshold:
            return "Unknown", min_distance
    
    if label_mapping is not None:
        predicted_label = next(
            name for name, idx in label_mapping.items()
            if idx == predicted_label
        )
    
    return predicted_label, min_distance

def recognize_from_distances(distances, threshold=None):
    label, distance = predict_label_from_distances(distances)

    if threshold is not None and distance > threshold:
            return "Unknown", distance

    return label, distance

def recognize_face_eigenface(
    test_face,
    mean_faces,
    eigenfaces,
    class_centroids,
    threshold=None
):
    # Step 1: Project face
    test_projection = project_face(test_face, mean_faces, eigenfaces)

    # Step 2: Compute distances
    distances = compute_distances_to_centroids(
        test_projection,
        class_centroids
    )

    # Step 3: Recognize
    label, distance = recognize_from_distances(distances, threshold)

    return label, distance, distances

def recognize_face_from_image(
    image_path,
    mean_faces,
    eigenfaces,
    class_centroids,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    threshold=None
):

    # Step 1: Preprocess image
    test_face = preprocess_face(
        image_path,
        TARGET_WIDTH=TARGET_WIDTH,
        TARGET_HEIGHT=TARGET_HEIGHT
    )

    # Step 2: Recognize
    return recognize_face_eigenface(
        test_face,
        mean_faces,
        eigenfaces,
        class_centroids,
        threshold
    )


# Utility function for prediction
from pathlib import Path

def predict_face_with_centroids(
    image_path,
    mean_faces,
    eigenfaces,
    class_centroids,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    label_mapping=None,
    threshold=None,
    display_plot=False
):

    test_face = preprocess_face(image_path, TARGET_WIDTH=TARGET_WIDTH, TARGET_HEIGHT=TARGET_HEIGHT, display_plot=display_plot)
    test_projection = project_face(test_face, mean_faces, eigenfaces)

    distances = compute_distances_to_centroids(
        test_projection,
        class_centroids
    )
    
    predicted_label, min_distance = predict_label_from_distances(distances, label_mapping=label_mapping)
    
    labels = list(distances.keys())
    values = list(distances.values())
    labels, values = zip(*sorted(zip(labels, values), key=lambda x: x[1]))
    
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    
    plt.subplot(2, 1, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Pred: {predicted_label}\nDist: {min_distance:.1f}")
    plt.axis("off")
    
    plt.subplot(2, 1, 2)
    plt.bar(labels, values, color="gray")
    
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=0.5, label='Threshold')
        if values[0] < threshold:
            plt.bar(labels[0], values[0])
        # else:
        #     plt.bar(labels[0], values[0], color="blue")
    else:
        plt.bar(labels[0], values[0])
            
    if label_mapping is not None:
        # For x-ticks
        x = np.arange(len(labels))
        names = [k for k, v in label_mapping.items() if v in labels]

        plt.xticks(x, names, rotation=45, ha="right") 
    else:
        plt.xticks(rotation=45, ha="right")
    plt.ylabel("Distance")
    plt.title("Distance to Centroids")

def predict_batch_label_display(
    test_image_list,
    mean_faces,
    eigenfaces,
    class_centroids,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    label_mapping=None,
    input_image_dir = Path('input_images')
    ):
    
    num_images = len(test_image_list)
    cols = 4
    rows = int(np.ceil(num_images / cols))


    plt.figure(figsize=(cols * 3, rows * 3))
    
    for idx, image_name in enumerate(test_image_list):

        test_image_path = input_image_dir / image_name

        # preprocess and predict
        test_face = preprocess_face(test_image_path, TARGET_WIDTH=TARGET_WIDTH, TARGET_HEIGHT=TARGET_HEIGHT)
        test_proj = project_face(test_face, mean_faces, eigenfaces)

        distances = compute_distances_to_centroids(
            test_proj,
            class_centroids
        )

        predicted_label, min_distance = predict_label_from_distances(distances, label_mapping=label_mapping)

        # load image again for display
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

        # subplot of prediction result
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Pred: {predicted_label}\nDist: {min_distance:.1f}")
        plt.axis("off")

def predict_batch_detailed_label_display(
    test_image_list,
    mean_faces,
    eigenfaces,
    class_centroids,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    threshold=None,
    label_mapping=None,
    input_image_dir = Path('input_images')
    ):
    
    num_images = len(test_image_list)
    rows = 2
    cols = num_images
    
    plt.figure(figsize=(4 * num_images, 6))
    
    for idx, image_name in enumerate(test_image_list):

        test_image_path = input_image_dir / image_name

        # preprocess and predict
        test_face = preprocess_face(test_image_path, TARGET_WIDTH=TARGET_WIDTH, TARGET_HEIGHT=TARGET_HEIGHT)
        test_proj = project_face(test_face, mean_faces, eigenfaces)

        distances = compute_distances_to_centroids(
            test_proj,
            class_centroids
        )

        predicted_label, min_distance = predict_label_from_distances(distances, threshold=threshold, label_mapping=label_mapping)

        labels = list(distances.keys())
        values = list(distances.values())
        labels, values = zip(*sorted(zip(labels, values), key=lambda x: x[1]))
        
        # load image again for display
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

        # Plot image
        img_pos = 0 * cols + idx + 1
        plt.subplot(rows, cols, img_pos)
        plt.imshow(image, cmap="gray")
        plt.title(f"Pred: {predicted_label}\nDist: {min_distance:.1f}")
        plt.axis("off")

        # Plot Bar Chart
        bar_pos = 1 * cols + idx + 1
        plt.subplot(rows, cols, bar_pos)
        plt.bar(labels, values, color="gray")
        if threshold is not None:
            plt.axhline(y=threshold, color='red', linestyle='--', linewidth=0.5, label='Threshold')
            if values[0] < threshold:
                plt.bar(labels[0], values[0])
            # else:
            #     plt.bar(labels[0], values[0], color="blue")
        else:
            plt.bar(labels[0], values[0])
            
        if label_mapping is not None:
            # For x-ticks
            x = np.arange(len(labels))
            names = [k for k, v in label_mapping.items() if v in labels]

            plt.xticks(x, names, rotation=45, ha="right") 
        else:
            plt.xticks(rotation=45, ha="right")
        
        plt.ylabel("Distance")
        plt.title("Distance to Centroids")

    plt.tight_layout()
    plt.show()