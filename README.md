# BiCoA-Net
Official repository for 'BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics'


 KinetiX.csv is the dataset we curated.


## Usage 
1. Download the model weights from the directory model_weights.
2. Prepare the input data, a csv file with the columns FASTA and smiles. (For example, input.csv)
3. Run with the following command
python inference.py --checkpoint path/to/your/model.pt --input input_data.csv --output predictions.csv --device cuda

