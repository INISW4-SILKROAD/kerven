import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS

# Enable debugging for more detailed error messages
app.debug = True

# Load embeddings file
dataset_root = "./static"
embeddings_path = f"{dataset_root}/embeddings11.npz"
data = np.load(embeddings_path)

# Extract arrays from the .npz file
gallery_embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
gallery_paths = data["image_paths"].astype(str)  # Convert to Python strings
gallery_labels = data["labels"] if "labels" in data else None
gallery_gids = data["gids"] if "gids" in data else None

# Load model and transformer
extractor = ViTExtractor.from_pretrained("vits16_dino")
extractor.eval()  # Set model to evaluation mode
transform, _ = get_transforms_for_pretrained("vits16_dino")

# Cosine similarity function
def cosine_similarity(x1, x2):
    return torch.nn.functional.cosine_similarity(x1, x2)

# Extract features for a given image
def extract_features(image):
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = extractor(image_transformed)
    return features

# Search endpoint
@app.route('/search', methods=['POST'])
def search_similar_images():
    try:
        if 'image_url' in request.json:
            image_url = request.json['image_url']
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(image_file).convert("RGB")
        else:
            return jsonify({"error": "No image or image URL provided"}), 400

        # Extract features from the image
        features_query = extract_features(image)

        # Calculate cosine similarity
        top_k = 5
        cosine_dists = cosine_similarity(features_query, gallery_embeddings)
        closest_indices_cosine = torch.topk(cosine_dists, top_k, dim=0).indices

        nearest_images_cosine = gallery_paths[closest_indices_cosine]
        nearest_distances_cosine = cosine_dists[closest_indices_cosine]
        nearest_gids_cosine = gallery_gids[closest_indices_cosine] if gallery_gids is not None else [""] * top_k

        # Prepare results
        results = []
        for img_path, dist, gid in zip(nearest_images_cosine, nearest_distances_cosine, nearest_gids_cosine):
            filename = os.path.basename(img_path)  # Extract filename from path
            results.append({"image_path": img_path, "similarity": dist.item(), "gid": int(gid), "filename": filename})

        return jsonify(results)

    except Exception as e:
        # Log the error for debugging purposes
        app.logger.error(f"Error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# Serve static files for images
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(dataset_root, filename)

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

