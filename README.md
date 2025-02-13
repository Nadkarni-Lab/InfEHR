# InfEHR: Resolving Clinical Uncertainty through Deep Geometric Learning on Electronic Health Records
This repository accompanies the preprint:  
**InfEHR: Resolving Clinical Uncertainty through Deep Geometric Learning on Electronic Health Records https://www.medrxiv.org/content/10.1101/2025.01.31.25321471v1)**  

## Usage Examples

### Build Attributed Graphs
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

---
