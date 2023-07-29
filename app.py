from flask import Flask, render_template, request, jsonify
import clip
import torch
from PIL import Image

WINNING_THRESHOLD = 0.65  # 75% confidence
STARTER_IMAGE = "cats.jpeg"
STARTER_ANSWER = "a cat"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(STARTER_IMAGE)).unsqueeze(0).to(device)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_image = request.files["file"]
        user_image = Image.open(user_image)

        user_image = preprocess(user_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # compare "user image" to "starter image"
            image_features = model.encode_image(image)
            user_image_features = model.encode_image(user_image)

            # cosine sim
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(
                image_features.cpu(), user_image_features.cpu()
            )

        return jsonify({"similarity": similarity.item(), "winning_threshold": WINNING_THRESHOLD})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)