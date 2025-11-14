# BiCoA-Net
Official repository for 'BiCoA-Net: An Interpretable Bidirectional Co-Attention Framework for Predicting Protein-Ligand Binding Kinetics'


 KinetiX.csv is the dataset we curated.


## Usage 
python inference.py \
    --checkpoint path/to/your/model.pt \
    --input input_data.csv \
    --output predictions.csv \
    --device cuda
