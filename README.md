# Synthetic-data-for-LDA-and-MLDA
Synthetic data generator for LDA
Synthetic data generator for MLDA

# HOW TO
## generator_lda.py
You can generate systhetic data for LDA  
`python3 generator_lda.py`  
- "tr.txt" is a training data for LDA
- "tr.z" is a label data (Topic propotion)  

You can generate test data for LDA  
`python3 generator_lda.py --test True`  
- "te.txt" is a test data for LDA
- "te.z" is a label data (Topic propotion)

## generator_mlda.py
You can generate systhetic data for MLDA  
`python3 generator_mlda.py`  
- "tr_w.txt" is a training data for MLDA (W1)
- "tr_f.txt" is a training data for MLDA (W2)
- "tr.z" is a label data (Topic propotion)  

You can generate test data for MLDA  
`python3 generator_mlda.py --test True`  
- "te_w.txt" is a test data for MLDA (W1)
- "te_f.txt" is a test data for MLDA (W2)
- "te.z" is a label data (Topic propotion)


# Note
When you do clustering by using LDA or MLDA  
**PLEASE Set MODE to True**  
`python3 generator_lda.py --mode True`  
`python3 generator_mlda.py --mode True`  
- mode==True : You can operate word distribution  
For example(Topic = 3,Words=10,Length of hist=6):  
[5 5 0 0 0 0 ] label is 0  
[0 0 6 4 0 0 ] label is 1  
[0 0 0 0 3 7 ] label is 2  
[2 8 0 0 0 0 ] label is 0  

**if not to set MODE to True**  
ARI cannot be measured because it is generated from uniform random numbers
