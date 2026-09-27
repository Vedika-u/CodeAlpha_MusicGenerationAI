import streamlit as st
import os
import glob
from utils import generate_random_music
from generate import generate_music

st.set_page_config(page_title="AI Music Generator", layout="wide")

st.title("🎹 AI Music Generator")
st.write("Generate music using LSTM Neural Networks or music theory rules!")

with st.sidebar:
    st.header("Settings")
    
    generation_mode = st.radio("Generation Mode", ["Demo Mode (Music Theory)", "AI Model (LSTM)"])
    
    st.subheader("Parameters")
    num_notes = st.slider("Number of Notes", min_value=10, max_value=500, value=100, step=10)
    
    temperature = 1.0
    if generation_mode == "AI Model (LSTM)":
        temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        st.info("Higher temperature = more creative/random. Lower = more predictable/safe.")
        
    st.header("Training Instructions")
    st.markdown("""
    **To train your own model:**
    1. Create a folder named `midi_songs` in the project root.
    2. Add some piano MIDI files (.mid) to the folder. You can find free MIDI files on sites like [freemidi.org](https://freemidi.org/) or [vgmusic.com](https://www.vgmusic.com/).
    3. Run the training script from the terminal:
       ```bash
       python train.py --epochs 50 --batch_size 64
       ```
    4. Once trained, the model will be saved in the `weights` folder, and you can use the AI Model generation mode!
    """)

st.header("Generate Music")

if st.button("Generate Music 🎵", use_container_width=True):
    with st.spinner("Generating music... This might take a moment."):
        output_file = None
        
        if generation_mode == "Demo Mode (Music Theory)":
            output_file = generate_random_music(num_notes=num_notes)
            st.success("Successfully generated demo music using music theory rules!")
            
        elif generation_mode == "AI Model (LSTM)":
            model_files = glob.glob("weights/*.pt")
            
            if not model_files:
                st.error("No trained model found! Please train a model first using `python train.py` or switch to Demo Mode.")
            else:
                model_path = 'weights/final_model.pt'
                if not os.path.exists(model_path):
                    model_path = model_files[-1]
                    
                st.info(f"Using model: {model_path}")
                output_file = generate_music(model_weights=model_path, num_notes=num_notes, temperature=temperature)
                
                if output_file:
                    st.success("Successfully generated music using AI!")
                else:
                    st.error("Failed to generate music. Make sure the data/notes.pkl file exists.")

        if output_file and os.path.exists(output_file):
            st.subheader("Download your track")
            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download MIDI File 📥",
                    data=f,
                    file_name="ai_generated_music.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
