import glob
import pickle
import os
import numpy as np
from music21 import converter, instrument, note, chord


def get_notes(midi_folder="midi_songs"):
    """Get all the notes and chords from the midi files in the directory."""
    notes = []

    if not os.path.exists(midi_folder):
        os.makedirs(midi_folder)
        print(f"Created {midi_folder} directory. Please add some MIDI files to it.")
        return notes

    files = glob.glob(f"{midi_folder}/*.mid")
    if not files:
        print(f"No MIDI files found in {midi_folder}.")
        return notes

    for file in files:
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}")
            notes_to_parse = None

            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Failed to parse {file}: {e}")

    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab, sequence_length=100):
    """Prepare the sequences used by the Neural Network."""
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((n, number) for number, n in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape for PyTorch: (samples, sequence_length, 1) and normalize
    network_input_np = np.array(network_input)
    network_input_reshaped = np.reshape(network_input_np, (n_patterns, sequence_length, 1))
    network_input_reshaped = network_input_reshaped / float(n_vocab)

    network_output_np = np.array(network_output)

    return network_input, network_input_reshaped, network_output_np
