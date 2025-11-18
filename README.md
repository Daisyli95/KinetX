# BiCoA-Net
Official repository for 'BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics (Under Reivew)'


 KinetiX.csv is the dataset we curated.

## First install the dependencies
git clone https://github.com/Daisyli95/KineticX.git
cd KineticX
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

## Usage 
1. Download the model weights from the directory model_weights.
2. Prepare the input data, a csv file with the columns FASTA and smiles. (For example, input.csv)
3. Run with the following command
python inference.py --checkpoint path/to/your/model.pt --input input_data.csv --output predictions.csv --device cuda

