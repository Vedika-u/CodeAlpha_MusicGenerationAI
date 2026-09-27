import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import get_notes, prepare_sequences
from model import create_network


def train_network(epochs=50, batch_size=64, midi_folder="midi_songs"):
    """Train a Neural Network to generate music."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Extracting notes from MIDI files...")
    notes = get_notes(midi_folder)

    if not notes:
        print("No notes found. Exiting.")
        return

    n_vocab = len(set(notes))
    print(f"Vocabulary size: {n_vocab}")

    print("Preparing sequences...")
    network_input, network_input_reshaped, network_output = prepare_sequences(notes, n_vocab)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(network_input_reshaped).to(device)
    y = torch.LongTensor(network_output).to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Creating network model...")
    model = create_network(n_vocab).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if not os.path.exists('weights'):
        os.makedirs('weights')

    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"weights/weights-improvement-{epoch+1:02d}-{avg_loss:.4f}.pt")

    print("Training finished! Saving final model...")
    torch.save(model.state_dict(), 'weights/final_model.pt')
    print("Model saved to weights/final_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--midi_folder", type=str, default="midi_songs")
    args = parser.parse_args()

    train_network(args.epochs, args.batch_size, args.midi_folder)
