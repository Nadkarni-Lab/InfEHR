# InfEHR: Resolving Clinical Uncertainty through Deep Geometric Learning on Electronic Health Records
This repository accompanies the preprint:  
**InfEHR: Resolving Clinical Uncertainty through Deep Geometric Learning on Electronic Health Records https://www.medrxiv.org/content/10.1101/2025.01.31.25321471v1)**  

## Usage Examples

### 1. Build Attributed Graphs
To construct patient graphs from the preprocessed data:
```bash
python src/build_graphs.py \
    --log-dir ./logs \
    --graph-tuples ./models/graph_ready_tuples_sem.bz2 \
    --embeddings ./node_embeddings/graph_ready_tuples_embeddings.bz2 \
    --time-embeddings ./models/time_embeddings_dict.pkl \
    --output-dir ./outputs/graphs \
    --d2v-model ./models/d2v_98291_17.169918439404636.pth \
    --run-name graph_construction \
    --semantic
```

### 2. Apply Labels to Graphs
To apply probabilistic labels to graphs:

```bash
python src/build_graphs/add_labels.py \
    --base-dir ./data \
    --graph-dir ./outputs/graphs \
    --labels-path ./data/patient_labels.pkl \
    --output-dir ./outputs/labeled_graphs \
    --log-dir ./logs \
    --batch-size 100 \
    --prefix labeled \
    --mrn-column MRN \
    --label-column prob_array
```

### 3. InfEHR GNN can be trained with and without supervision:

#### Self-supervised:
For self-supervised pre-training using custom loss:
```bash
python src/train_unlabeled.py \
    --log-dir ./logs \
    --model-save-dir ./models \
    --graph-dir ./outputs/graphs \
    --graph-mapping ./data/mrn_to_file.json \
    --sim-weight 0.5 \
    --var-weight 3.0 \
    --cov-weight 2.0 \
    --mi-weight 1.0 \
    --input-dim 464 \
    --hidden-dim 256 \
    --embedding-dim 128 \
    --num-heads 2 \
    --pool-ratio 0.2 \
    --epochs 1000 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --accumulation-steps 64 \
    --batch-size 1 \
    --device cuda
```
#### Weakly supervised:
For training with probabilistic labels:
```bash
python src/train.py \
    --log-dir ./logs \
    --model-save-dir ./models \
    --graph-dir ./outputs/labeled_graphs \
    --graph-mapping ./data/mrn_to_file.json \
    --input-dim 464 \
    --hidden-dim 256 \
    --embedding-dim 128 \
    --num-heads 2 \
    --pool-ratio 0.2 \
    --num-classes 2 \
    --epochs 150 \
    --learning-rate 0.001 \
    --weight-decay 1e-4 \
    --accumulation-steps 64 \
    --batch-size 1 \
    --custom-loss-epoch 1 \
    --weight-lr 0.0001 \
    --device cuda
```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{kauffman2025infehr,
   title={InfEHR: Resolving Clinical Uncertainty through Deep Geometric Learning on Electronic Health Records},
   author={Kauffman, Justin and others},
   journal={medRxiv},
   year={2025},
   doi={10.1101/2025.01.31.25321471}
}
