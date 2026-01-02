# BiCoA-Net

Official repository for **"BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics"** (Under Review)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Weights](#model-weights)
- [License](#license)

## 🔬 Overview

BiCoA-Net is a deep learning framework designed to predict protein-ligand binding kinetics with enhanced interpretability through bidirectional co-attention mechanisms.

## 📊 Dataset

**KinetX.csv** - A curated dataset for protein-ligand binding kinetics prediction.
- Contains protein FASTA sequences and ligand SMILES strings
- Includes experimental kinetic measurements

**casestudy.csv**-The dataset contain information about the four held-out targets, with no overlap with KinetX.csv. 


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8 (for GPU support)
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Daisyli95/KineticX.git
cd KinetX
```

2. **Install PyTorch (CUDA 11.8)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: For other CUDA versions or CPU-only installation, visit [PyTorch official website](https://pytorch.org/get-started/locally/)

3. **Install other dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

1. **Download model weights**
   
   Download the pre-trained model from the `model_weights/` directory.

2. **Prepare input data**
   
   Create a CSV file with the following columns:
   - `FASTA`: Protein sequence in FASTA format
   - `smiles`: Ligand structure in SMILES format
   
   Example (`input.csv`):
```csv
   FASTA,smiles
   MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,CC(C)Cc1ccc(cc1)C(C)C(=O)O
```

3. **Run inference**
```bash
python inference.py \
    --checkpoint path/to/your/model.pt \
    --input input_data.csv \
    --output predictions.csv \
    --device cuda
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint file | Required |
| `--input` | Path to input CSV file | Required |
| `--output` | Path to save predictions | `predictions.csv` |
| `--device` | Device to use (`cuda` or `cpu`) | `cuda` |

## 📦 Model Weights

Pre-trained model checkpoints are available in the `model_weights/` directory via Hugging Face.


## 📄 License
MIT license

## 📧 Contact

For questions or issues, please open an issue on GitHub.

