#!/usr/bin/env python3
"""
Simple pKoff Prediction Script
Load model checkpoint, read FASTA + SMILES from CSV, output predicted pKoff values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings
import esm
from transformers import AutoTokenizer, AutoModel
import math

# RDKit for molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================== Molecular Descriptors ========================

def compute_molecular_descriptors(smiles):
    """Compute RDKit molecular descriptors"""
    if not RDKIT_AVAILABLE:
        return np.zeros(200)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200)
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.RingCount(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=182)
        fp_array = np.array(fp)
        
        full_desc = np.concatenate([descriptors, fp_array])
        return full_desc
        
    except Exception as e:
        print(f"  Error computing descriptors for SMILES: {e}")
        return np.zeros(200)


# ======================== Embedding Functions ========================

class MolFormerEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading MolFormer...")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", use_safetensors=True)
        self.model = self.model.to(device).eval()
        print("✓ MolFormer loaded")
    
    def embed_smiles(self, smiles, max_length=512):
        try:
            with torch.no_grad():
                inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(768), False


class ESM2Embedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device='cuda'):
        self.device = device
        print(f"Loading ESM-2: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print("✓ ESM-2 loaded")
        
    def embed_protein(self, sequence, max_length=1024):
        try:
            with torch.no_grad():
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                data = [("protein", sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[33])
                return results["representations"][33].mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(1280), False


def normalize_embeddings(embeddings):
    """Normalize embeddings (subtract mean, divide by std)"""
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (embeddings - mean) / std


# ======================== Model Architecture ========================

class MultiHeadCoAttentionWithTemp(nn.Module):
    """Multi-head co-attention with temperature scaling"""
    def __init__(self, d_model, n_heads=8, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        self.q_lig = nn.Linear(d_model, d_model)
        self.k_lig = nn.Linear(d_model, d_model)
        self.v_lig = nn.Linear(d_model, d_model)
        
        self.q_prot = nn.Linear(d_model, d_model)
        self.k_prot = nn.Linear(d_model, d_model)
        self.v_prot = nn.Linear(d_model, d_model)
        
        self.out_lig = nn.Linear(d_model, d_model)
        self.out_prot = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout * 0.5)
        self.scale = math.sqrt(self.d_k) * self.temperature
    
    def forward(self, lig, prot):
        batch_size = lig.size(0)
        
        def reshape(x):
            return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Ligand attends to protein
        q_l = reshape(self.q_lig(lig.unsqueeze(1)))
        k_p = reshape(self.k_prot(prot.unsqueeze(1)))
        v_p = reshape(self.v_prot(prot.unsqueeze(1)))
        
        attn_l = torch.matmul(q_l, k_p.transpose(-2, -1)) / self.scale
        attn_l = F.softmax(attn_l, dim=-1)
        attn_l = self.attn_dropout(attn_l)
        
        out_l = torch.matmul(attn_l, v_p)
        out_l = out_l.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_l = self.out_lig(out_l).squeeze(1)
        
        # Protein attends to ligand
        q_p = reshape(self.q_prot(prot.unsqueeze(1)))
        k_l = reshape(self.k_lig(lig.unsqueeze(1)))
        v_l = reshape(self.v_lig(lig.unsqueeze(1)))
        
        attn_p = torch.matmul(q_p, k_l.transpose(-2, -1)) / self.scale
        attn_p = F.softmax(attn_p, dim=-1)
        attn_p = self.attn_dropout(attn_p)
        
        out_p = torch.matmul(attn_p, v_l)
        out_p = out_p.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_p = self.out_prot(out_p).squeeze(1)
        
        return out_l, out_p


class RefinedCoAttentionBlock(nn.Module):
    """Refined co-attention block with residual connections"""
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadCoAttentionWithTemp(d_model, n_heads, dropout)
        
        self.norm1_lig = nn.LayerNorm(d_model)
        self.norm1_prot = nn.LayerNorm(d_model)
        self.norm2_lig = nn.LayerNorm(d_model)
        self.norm2_prot = nn.LayerNorm(d_model)
        
        self.ff_lig = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.ff_prot = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lig, prot):
        # Co-attention with residuals
        attn_lig, attn_prot = self.attention(lig, prot)
        lig = self.norm1_lig(lig + self.dropout(attn_lig))
        prot = self.norm1_prot(prot + self.dropout(attn_prot))
        
        # Feedforward with residuals
        lig = self.norm2_lig(lig + self.ff_lig(lig))
        prot = self.norm2_prot(prot + self.ff_prot(prot))
        
        return lig, prot


class RefinedProteinLigandModel(nn.Module):
    """
    Refined protein-ligand binding prediction model with:
    - Bidirectional co-attention
    - Multi-scale interaction modeling
    - Molecular descriptor integration
    """
    def __init__(self, smiles_dim=768, protein_dim=1280, descriptor_dim=200,
                 d_model=512, n_blocks=4, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Input projections
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.descriptor_proj = nn.Sequential(
            nn.Linear(descriptor_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Co-attention blocks
        self.coattn_blocks = nn.ModuleList([
            RefinedCoAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        
        # Interaction fusion
        self.interaction_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, smiles_emb, protein_emb, descriptors):
        # Project inputs
        lig = self.smiles_proj(smiles_emb)
        prot = self.protein_proj(protein_emb)
        desc = self.descriptor_proj(descriptors)
        
        # Apply co-attention blocks
        for block in self.coattn_blocks:
            lig, prot = block(lig, prot)
        
        # Fuse interactions
        interaction = torch.cat([lig, prot], dim=-1)
        fused = self.interaction_fusion(interaction)
        
        # Combine with descriptors
        combined = torch.cat([fused, desc], dim=-1)
        
        # Predict
        output = self.predictor(combined)
        return output.squeeze(-1)


# ======================== Prediction Function ========================

def predict_pkoff(checkpoint_path, input_csv, output_csv, device='cuda'):
    """
    Load model and predict pKoff values
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        input_csv: Path to input CSV with 'FASTA' and 'smiles' columns
        output_csv: Path to save predictions
        device: 'cuda' or 'cpu'
    """
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*70}")
    print("PKOFF PREDICTION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    normalization = checkpoint['normalization']
    
    print("✓ Checkpoint loaded")
    
    # Initialize model
    model = RefinedProteinLigandModel(
        smiles_dim=config['smiles_dim'],
        protein_dim=config['protein_dim'],
        descriptor_dim=config['descriptor_dim'],
        d_model=config['d_model'],
        n_blocks=config['n_blocks'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    # Initialize embedders
    molformer = MolFormerEmbedder(device=device)
    esm2 = ESM2Embedder(device=device)
    
    # Load input data
    print(f"\nLoading input data...")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} samples")
    
    # Check required columns
    if 'FASTA' not in df.columns or 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain 'FASTA' and 'smiles' columns")
    
    # Compute embeddings
    print("\nComputing SMILES embeddings...")
    smiles_embeddings = []
    for smiles in tqdm(df['smiles'].tolist(), desc="SMILES"):
        emb, _ = molformer.embed_smiles(smiles)
        smiles_embeddings.append(emb)
    smiles_embeddings = normalize_embeddings(np.array(smiles_embeddings))
    
    print("\nComputing protein embeddings...")
    protein_embeddings = []
    for fasta in tqdm(df['FASTA'].tolist(), desc="Proteins"):
        emb, _ = esm2.embed_protein(fasta)
        protein_embeddings.append(emb)
    protein_embeddings = normalize_embeddings(np.array(protein_embeddings))
    
    print("\nComputing molecular descriptors...")
    mol_descriptors = []
    for smiles in tqdm(df['smiles'].tolist(), desc="Descriptors"):
        desc = compute_molecular_descriptors(smiles)
        mol_descriptors.append(desc)
    mol_descriptors = np.array(mol_descriptors)
    
    # Normalize descriptors using training statistics
    desc_mean = normalization['desc_mean']
    desc_std = normalization['desc_std']
    mol_descriptors = (mol_descriptors - desc_mean) / desc_std
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(len(df)), desc="Predicting"):
            smiles_feat = torch.FloatTensor(smiles_embeddings[i:i+1]).to(device)
            protein_feat = torch.FloatTensor(protein_embeddings[i:i+1]).to(device)
            desc_feat = torch.FloatTensor(mol_descriptors[i:i+1]).to(device)
            
            # Predict (normalized)
            pred_normalized = model(smiles_feat, protein_feat, desc_feat)
            
            # Denormalize to original scale
            pred_original = pred_normalized.cpu().item() * normalization['std'] + normalization['mean']
            predictions.append(pred_original)
    
    # Add predictions to dataframe
    df['predicted_pkoff'] = predictions
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Predict pKoff values from FASTA + SMILES')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with FASTA and smiles columns')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Run predictions
    predict_pkoff(
        checkpoint_path=args.checkpoint,
        input_csv=args.input,
        output_csv=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
