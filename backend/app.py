

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import uuid
# from measurement_logic import analyze_video

# app = Flask(__name__)
# CORS(app)

# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

# @app.route('/process_video', methods=['POST'])
# def process_video_endpoint():
#     if 'video' not in request.files or 'frameWidth' not in request.form:
#         return jsonify({'error': 'Missing video file or frame width'}), 400

#     video_file = request.files['video']
#     frame_width = float(request.form['frameWidth'])
    
#     filename = str(uuid.uuid4()) + '.mp4'
#     video_path = os.path.join('uploads', filename)
#     video_file.save(video_path)

#     try:
#         measurements, landmarks, frame_dims = analyze_video(video_path, frame_width)
        
#         response_data = {
#             "measurements": measurements,
#             "landmarks": landmarks,
#             "frameDimensions": frame_dims
#         }
        
#         return jsonify(response_data)

#     except Exception as e:
#         app.logger.error(f"An error occurred: {e}")
#         return jsonify({'error': str(e)}), 500

#     finally:
#         if os.path.exists(video_path):
#             os.remove(video_path)

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from measurement_logic import analyze_video
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    """
    Endpoint to process a video file for optical measurements.
    Expects a multipart form with 'video' and 'frame_width_mm'.
    """
    # --- 1. Validate Input ---
    if 'video' not in request.files:
        app.logger.warning("Request received without video file.")
        return jsonify({'error': 'Missing video file'}), 400

    if 'frame_width_mm' not in request.form:
        app.logger.warning("Request received without frame_width_mm.")
        return jsonify({'error': 'Missing frame_width_mm parameter'}), 400

    video_file = request.files['video']
    
    try:
        # Convert frame_width_mm to a float for calculation
        frame_width_mm = float(request.form['frame_width_mm'])
    except ValueError:
        app.logger.error("Invalid format for frame_width_mm. Could not convert to float.")
        return jsonify({'error': 'frame_width_mm must be a valid number'}), 400

    # --- 2. Save Video File ---
    # Create a unique filename to avoid conflicts
    filename = str(uuid.uuid4()) + '.mp4'
    video_path = os.path.join('uploads', filename)
    video_file.save(video_path)
    app.logger.info(f"Video saved to {video_path}")

    # --- 3. Analyze Video ---
    try:
        # Call the analysis function with the required frame_width_mm argument
        app.logger.info(f"Analyzing video with frame width: {frame_width_mm}mm")
        measurements, landmarks, frame_dims = analyze_video(video_path, frame_width_mm=frame_width_mm)
        
        # Prepare the successful response
        response_data = {
            "measurements": measurements,
            "landmarks": landmarks,
            "frameDimensions": frame_dims
        }
        
        return jsonify(response_data)

    except Exception as e:
        # Log the full exception for easier debugging
        app.logger.error(f"Analysis failed for {video_path}: {e}", exc_info=True)
        return jsonify({'error': f"Analysis Failed: {e}"}), 500

    finally:
        # --- 4. Cleanup ---
        # Ensure the temporary video file is always deleted
        if os.path.exists(video_path):
            os.remove(video_path)
            app.logger.info(f"Cleaned up temporary file: {video_path}")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)