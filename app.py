from flask import Flask, request, render_template, send_file, jsonify
import os
from music21 import *
from fractions import Fraction
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from schubert import *


app = Flask(__name__)

@app.route('/')
@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/schubert')
def schubert():
    return render_template('schubert.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/otherprojects')
def otherprojects():
    return render_template('otherprojects.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join("uploads", filename), as_attachment=True)


@app.route('/process', methods=['POST'])
def process_file():
    """
    Handles a file upload, processes a MusicXML (.mxl) file using a custom function,
    and returns a JSON response with a download link for the modified file.
    """

    try:
        # Check if a file was sent in the request
        if 'file' not in request.files:
            return jsonify(success=False, message="No file uploaded.")  # previously "Aucun fichier envoyé."

        uploaded_file = request.files['file']

        # Ensure the uploaded file has a .mxl extension
        if not uploaded_file.filename.lower().endswith('.mxl'):
            return jsonify(success=False, message="Please send a MusicXML (.mxl) file only!")
        
        # Retrieve the slider value from the form (default to 4 if invalid)
        nbarline = int(request.form.get("nbarline", 4))

        # Save the uploaded file to the 'uploads' folder
        input_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(input_path)

        # Try to process the file using the modify_file_with_schubert function
        try:
            output_path = modify_file_with_schubert(input_path, nbarline)
            
        except Exception as e:
            print("SERVER ERROR:", e)  # ✅ Logs the exact error in the terminal
            return jsonify(success=False, message="Error in the modification function")

        # Check if the modification function returned a valid path
        if output_path is None:
            return jsonify(success=False, message="Invalid file or key not found.")

        # If everything worked, return a JSON response with a download link
        return jsonify(
            success=True,
            download_url=f"/download/{os.path.basename(output_path)}"
        )

    except Exception as e:
        print("SERVER ERROR:", e) 
        return jsonify(success=False, message="Internal server error.")


def modify_file_with_schubert(filepath, nbarline=6, k=8):
    """
    Modifies a MusicXML (.mxl) file using a custom music generation model.

    Parameters:
    - filepath: path to the input MusicXML file
    - nbarline: number of bars per line for generation (default 6)
    - k: top-k sampling parameter for the music generation (default 8)

    Returns:
    - output_path: path to the modified MusicXML file
    """
    
    # Extract the musical key and the number of hands (nhand) from the score
    key_score, nhand = extract_key_and_nhand_from_score(filepath)
    print('test_1')  # Debug print to trace execution

    # Load the music generation model for the given number of hands
    model, token_to_id, id_to_token = load_model(nhand)
    print('test_2')

    # Convert the music file into a sequence of tokens for processing
    tokens = load_music_sheet(filepath, nhand)
    print('test_3')

    # Prepare the output file path by appending "_modified" before the extension
    output_path = filepath.replace('.mxl', '_modified.mxl')
    print('test_4') 

    # Attempt generation and writing up to 20 times in case of errors
    i = 0
    while i < 20:
        i += 1
        try:
            # Generate new music tokens using the model
            generated_tokens = generation(tokens, nbarline, model, token_to_id, id_to_token, k)
            print("test 5 : généré")
            
            # Write the generated tokens back into a MusicXML file
            write_music_sheet(generated_tokens, nhand, key_score, output_path)
            break  # Exit the loop if successful

        except Exception as e:
            # If an error occurs, log it and retry
            print(f"An error occurred: {e}. Restarting...")
    
    # Return the path to the modified MusicXML file
    return output_path


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)

    app.run(host="0.0.0.0", debug=True)
