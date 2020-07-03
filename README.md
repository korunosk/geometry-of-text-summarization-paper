# Geometry of Text Summarization

## Requirements

    more-itertools==8.4.0
    orjson==3.2.0
    numpy==1.18.0
    scipy==1.5.0
    matplotlib==3.2.2
    seaborn==0.10.1
    scikit-learn==0.23.1
    tensorflow==1.15
    torch==1.5.1
    torchvision==0.6.1
    bert-serving-client==1.10.0
    transformers==3.0.0
    gensim==3.8.3
    nltk==3.5
    py-rouge==1.1
    geomloss==0.2.3
    POT==0.7.0
    ray==0.8.6

The CUDA driver on 037 is older. Execute:

    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

## Directory structure

Datasets:

`base_data_dir/dataset_id.json`

Data:

`base_data_dir/data/dataset_id/item.npy`

Topics encoded:

`base_data_dir/embeddings/embedding_method/dataset_id/topic_id.json`

Topic items encoded:

`base_data_dir/embeddings/embedding_method/dataset_id/topic_id/item.npy`

Models:

`base_data_dir/models/embedding_method/dataset_id/model.pt`

> **Note**:
>
> Some of the embedding methods (ex. BERT) have additional subdirectory denoting the hidden state used to retrieve the representations.
> 
> Ex:
> `base_data_dir/embeddings/embedding_method/dataset_id/layer/topic_id.json`

## Start bert-as-service
    
    # Activate environment
    export PATH=/opt/anaconda3/bin:$PATH
    source activate dlab
    
    # Install dependencies
    pip install tensorflow==1.15
    pip install bert-serving-server==1.10.0
    
    # Get pretrained BERT model
    wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
    mkdir bert
    unzip uncased_L-12_H-768_A-12 -d bert

    # BERT word-level embeddings
    screen -r [id]
    bert-serving-start -model_dir=bert -port 5557 -port_out 5558 -num_worker=8 -max_seq_len=NONE -max_batch_size=64 -pooling_strategy=NONE -show_tokens_to_client
    # BERT sentence-level embeddings
    screen -r [id]
    bert-serving-start -model_dir=bert -port 5557 -port_out 5558 -num_worker=8 -max_seq_len=NONE -max_batch_size=64

## Monitor NVIDIA GPU unit

    nvidia-smi
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv -l 5

## Scripts

- `main_data.py`:           Generates train datasets
- `main_encoding.py`:       Encodes topic items
- `main_experiments.py`:    Executes baseline experiments
- `main_lsa.py`:            Generates LSA embeddings
- `main_rouge.py`:          Generater ROUGE scores for documents' sentences
- `main_split.py`:          Exports embedded items
- `main_training.py`:       Trains models

## Nomenclature

Common variable names

- dataset_ids
- topic_ids
- document, document_embs
- summary, summary_embs, summary_ids
- pyr_scores
- indices
