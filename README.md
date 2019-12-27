# Synthetic-data-for-LDA-and-MLDA
Synthetic data generator for LDA and MLDA

# HOW TO
## generator_lda.py
You can generate systhetic data for LDA  
`python3 generator_lda.py`  

## generator_mlda.py
You can generate systhetic data for MLDA  
`python3 generator_lda.py`  

# Note
When you do clustering by using LDA or MLDA  
**PLEASE Set MODE to True**
`python3 generator_lda.py --mode True`  

**if not to set MODE to True**  
ARI cannot be measured because it is generated from uniform random numbers
