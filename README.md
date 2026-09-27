# AI Music Generator

A complete AI Music Generation project using LSTM Neural Networks (TensorFlow/Keras) and music21 for MIDI processing, wrapped in a beautiful Streamlit user interface.

## Features
- **Streamlit Web UI:** Easy-to-use interface for music generation.
- **LSTM Model:** Stacked LSTM architecture for sequence prediction.
- **Demo Mode:** Generate music using music theory rules without needing a pre-trained model.
- **MIDI Processing:** Extract notes and chords from MIDI files and convert generated sequences back to playable MIDI files.
- **Customizable Generation:** Adjust the number of notes and the 'temperature' (creativity) of the generated music.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the App
To start the Streamlit interface, run:
```bash
streamlit run app.py
```

### Demo Mode
By default, you can use the **Demo Mode** in the app to generate music based on music theory rules. This works out-of-the-box without any training required!

### Training Your Own AI Model
If you want the AI to learn from existing MIDI files and generate its own sequences:

1. Create a directory named `midi_songs` in the root of the project.
2. Place your `.mid` training files in the `midi_songs` folder. (You can find free datasets online, such as classic piano pieces from [FreeMidi](https://freemidi.org/)).
3. Run the training script:
   ```bash
   python train.py --epochs 50 --batch_size 64
   ```
4. Once training completes, models will be saved in the `weights/` directory.
5. Relaunch the app (`streamlit run app.py`) and select **AI Model (LSTM)** in the sidebar to generate music!

## Technologies Used
- Python 3
- TensorFlow / Keras (LSTM modeling)
- Streamlit (Web interface)
- music21 (MIDI file parsing and generation)
- NumPy (Data manipulation)

## License
MIT License.
