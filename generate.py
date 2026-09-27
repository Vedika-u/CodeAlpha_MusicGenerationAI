import pickle
import numpy as np
import torch
from music21 import instrument, note, stream, chord
from model import create_network
from preprocess import prepare_sequences
from utils import create_midi_from_notes


def generate_music(model_weights='weights/final_model.pt', num_notes=100, temperature=1.0):
    """Generate a piano midi file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with open('data/notes.pkl', 'rb') as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        print("Error: data/notes.pkl not found. Please train the model first.")
        return None

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, network_input_reshaped, _ = prepare_sequences(notes, n_vocab)

    model = create_network(n_vocab).to(device)
    try:
        model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

    model.eval()

    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, n) for number, n in enumerate(pitchnames))
    pattern = list(network_input[start])
    prediction_output = []

    print("Generating notes...")
    with torch.no_grad():
        for note_index in range(num_notes):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            input_tensor = torch.FloatTensor(prediction_input).to(device)

            prediction = model(input_tensor)[0].cpu().numpy()

            # Apply temperature scaling
            prediction = np.log(prediction + 1e-7) / temperature
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)

            index = np.random.choice(len(prediction), p=prediction)
            result = int_to_note[index]
            prediction_output.append(result)

            pattern.append(index)
            pattern = pattern[1:]

    output_file = create_midi_from_notes(prediction_output, "generated_output.mid")
    return output_file
