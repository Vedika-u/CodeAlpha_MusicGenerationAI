import os
import random
import pickle
from music21 import note, chord, instrument, stream, scale, pitch
import numpy as np

def generate_random_music(num_notes=100):
    """Generate random music based on music theory (demo mode)"""
    # Simple C major scale
    s = scale.MajorScale('C')
    pitches = s.getPitches('C4', 'C6')
    
    output_notes = []
    offset = 0
    
    for _ in range(num_notes):
        # 80% chance of a single note, 20% chance of a chord
        if random.random() < 0.2:
            # Create a chord
            p = random.choice(pitches)
            c = chord.Chord([p, p.transpose('M3'), p.transpose('P5')])
            c.offset = offset
            output_notes.append(c)
        else:
            p = random.choice(pitches)
            n = note.Note(p)
            n.offset = offset
            n.quarterLength = random.choice([0.25, 0.5, 1.0])
            output_notes.append(n)
            
        offset += random.choice([0.25, 0.5])
        
    midi_stream = stream.Stream(output_notes)
    
    output_path = "demo_output.mid"
    midi_stream.write('midi', fp=output_path)
    return output_path

def create_midi_from_notes(prediction_output, output_filename="output.mid"):
    """ Convert the prediction output to a MIDI file """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    return output_filename
    
def load_vocab():
    if os.path.exists('data/notes.pkl'):
        with open('data/notes.pkl', 'rb') as f:
            notes = pickle.load(f)
        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(set(notes))
        return notes, pitchnames, n_vocab
    return None, None, None
