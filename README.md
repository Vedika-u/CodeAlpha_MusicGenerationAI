# CodeAlpha Artificial Intelligence Internship
## Task 3: AI Music Generation with Deep Learning (LSTM) & Streamlit

**Intern:** Vedika Utturwar  
**Student ID:** CA/DF1/308779  
**Domain:** Artificial Intelligence  
**Batch:** 1st October 2026 to 30th October 2026  

---

## 📌 Project Overview
A complete AI Music Generation system using **PyTorch LSTM Neural Networks** and **music21** for MIDI processing, wrapped in an interactive **Streamlit** user interface.

## ✨ Features
- **Streamlit Web UI:** Intuitive, modern interface for music generation and playback downloads.
- **PyTorch LSTM Architecture:** Multi-layer stacked LSTM network for sequential musical note and chord prediction.
- **Demo Mode:** Instant music generation based on algorithmic music theory rules without requiring pre-trained weights.
- **Customizable Generation:** Real-time controls for sequence length (number of notes) and temperature (creativity factor).
- **MIDI Processing:** End-to-end extraction and reconstruction of polyphonic chords and single notes.
- **One-Click Launcher:** Run effortlessly on Windows via `run.bat`.

## 🛠️ Technologies Used
- **Python 3**
- **PyTorch (`torch`)**: Deep learning LSTM sequence modeling
- **Streamlit**: Web interface & interactive audio controls
- **music21**: MIDI file parsing, note stream processing, and MIDI generation
- **NumPy & Pandas**: Data manipulation and preprocessing

## 🚀 Installation

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/Vedika-u/CodeAlpha_MusicGenerationAI.git
   cd CodeAlpha_MusicGenerationAI
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the App
- **Option 1 (Windows):** Double-click `run.bat`
- **Option 2 (CLI):**
  ```bash
  streamlit run app.py
  ```

### Demo Mode
By default, choose **Demo Mode (Music Theory)** in the sidebar to generate custom `.mid` tracks instantly.

### Training Your Own AI Model
To train the neural network on your own collection of MIDI songs:
1. Create a `midi_songs/` folder in the project root.
2. Add your favorite piano `.mid` files to `midi_songs/`.
3. Run the training script:
   ```bash
   python train.py --epochs 50 --batch_size 64
   ```
4. Trained checkpoints are saved under `weights/`. Switch the app to **AI Model (LSTM)** mode to generate novel music!

## 📂 Project Structure
```text
CodeAlpha_MusicGenerationAI/
├── app.py              # Streamlit web application
├── model.py            # PyTorch LSTM network architecture
├── train.py            # Training pipeline
├── generate.py         # AI inference and note generator
├── preprocess.py       # MIDI feature extraction and tokenization
├── utils.py            # Music theory demo synthesizer
├── requirements.txt    # Project dependencies
├── run.bat             # One-click Windows startup script
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
```

## 📄 License
MIT License.

