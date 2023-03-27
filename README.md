# Review-of-ReLSO

## Overview of the atricle

Powerful language models and advances in protein research have helped us learn more about protein sequences and create lots of useful data. The article introduces a new method called Regularized Latent Space Optimization (ReLSO) that uses this knowledge to better understand and predict the features of proteins. ReLSO is an efficient way to find the best protein sequences by exploring the connections between sequences and their functions. It has been tested on several protein datasets and showed better results than other methods, making it easier to find high-quality protein sequences.

### Problem
The main problem solved by this article is finding a better way to design proteins by exploring the huge number of possible sequences.

`Why is this a challenge?`
1. enormous size of the search space (A small protein of 30 residues translates into a total search space of $10^{38}$ — far beyond the reach of modern 
high-throughput screening technologies.)

2. epistasis (complex interactions between amino acids) makes it difficult to predict the effect of small changes in the sequence on its properties

### Method
Regularized Latent Space Optimization (ReLSO)

\
ReLSO is a transformer-based autoencoder that features a highly structured latent space and is trained to jointly generate protein sequences and predict their fitness.

\
`Main advantage`: it helps create proteins sequences with desired properties without having to explore every possible sequence.

\
ReLSO built based on JT-AE model by including additional regularizations (negative sampling and interpolation losses).

\
<img width="400" alt="Screen Shot 2023-03-26 at 11 49 16 PM" src="https://user-images.githubusercontent.com/89158696/227847495-950bc57b-6229-4a3b-a179-c0e30075f08e.png">

### Differences with other model
ReLSO (Regularized Latent Space Optimization) builds upon JT-AE (Jointly Trained Autoencoder) and introduces specific regularization techniques to improve the optimization process. The two key regularization methods used in ReLSO are:

\
`Negative sampling` is used to reshape the latent fitness landscape to be pseudoconvex, creating an implicit trust region and a natural stopping criterion for latent space optimization.

\
`Interpolation losses` help maintain a smooth transition between points in the latent space, ensuring that small changes in the latent representation lead to small changes in the output.

\
An inherent weakness of JT-AE, is that often a monotonic function is learned by the auxiliary network. Such a function lacks any stopping 
criterion when used for latent space optimization. To address this, we reshape the fitness function such that the global maximum lie in/near the training 
data.

\
In summary, while JT-AE focuses on jointly training an autoencoder with a fitness prediction network, ReLSO extends this concept by introducing regularization techniques to improve the optimization process in the latent space.

## Architecture
<img width="760" alt="Screen Shot 2023-03-26 at 12 27 20 PM" src="https://user-images.githubusercontent.com/89158696/227793235-16453185-2940-4346-a381-f2a5fbf8bea4.png">

\
`Model input`: a set of protein sequences and their fitness measurements

`Model output`: the reconstructed protein sequences and the predicted fitness values for the input sequences.

### Pseudocode:
Initialize ReLSO model with encoder $f_{\theta}$, decoder $g_{theta}$, and fitness prediction network $h_{\theta}$

for each epoch:

    for each batch of protein sequences (x) and fitness values (y):

        # Embedding, Transformer, Pooling, and Bottleneck (JT-AE Encoder)
        1. Convert protein sequence (x) into token-level representation using Embedding
        
        2. Process token-level representation using Transformer layers
        
        3. Obtain sequence-level representation using attention-based Pooling mechanism
        
        4. Generate low-dimensional latent representation (z) using Bottleneck layer

        # JT-AE (Decoder and Fitness Prediction Network)
        5. Reconstruct protein sequence using Decoder
        
        6. Predict fitness using fitness prediction network

        # Compute Losses
        7. Calculate reconstruction loss between original and reconstructed protein sequences
        
        8. Calculate fitness prediction loss between true and predicted fitness values

        # Negative Sampling
        9. Generate negative samples (zn) by sampling high-norm regions around real latent points
        
        10. Assign low fitness values to negative samples and include them in the fitness prediction loss

        # Interpolation Losses
        11. Enforce smoothness on the decoder by adding interpolation loss to the objective function

        # Update Objective Function
        12. Combine reconstruction loss, fitness prediction loss, negative sampling loss, and interpolation loss
        
        13. Update model parameters using gradients from the combined loss

<img width="400" alt="Screen Shot 2023-03-26 at 9 21 07 PM" src="https://user-images.githubusercontent.com/89158696/227843535-dc26f652-80b7-4cff-8787-0071086527cf.png">

### Optimization
<img width="983" alt="Screen Shot 2023-03-27 at 12 26 57 AM" src="https://user-images.githubusercontent.com/89158696/227849019-784c0c52-2944-45e4-9210-f66617984358.png">

<img width="558" alt="Screen Shot 2023-03-27 at 12 27 10 AM" src="https://user-images.githubusercontent.com/89158696/227849026-24472c1f-abea-41b4-916a-cecc906ebc15.png">

<img width="325" alt="Screen Shot 2023-03-27 at 12 27 26 AM" src="https://user-images.githubusercontent.com/89158696/227849044-02a64004-6acc-4c17-ac56-c7afaf2ffcb0.png">

## Results

<img width="726" alt="Screen Shot 2023-03-27 at 12 28 23 AM" src="https://user-images.githubusercontent.com/89158696/227848806-25d71d0d-1b80-438a-ae8b-b0f3642179a9.png">

## Critical Analysis

### What could have been developed further?
1. More diverse and larger training datasets: The model's performance depends on the quality and diversity of the training data. Including more diverse protein sequences and a wider range of protein properties could improve generalization and optimization performance.

2. Improved regularization techniques: Exploring new regularization strategies or refining existing ones.

3. Real-world validation: To ensure the effectiveness of the optimized protein sequences, it is crucial to validate the results with experimental data.

## Questions
`Question1:` Why did the results show like this?

`Question2:` What advantage does the Transformer architecture provide when used in the fitness prediction model?

## Code demonstration

### Preparation
```
# clone project   
git clone https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers.git

# with conda
conda env create -f relso_env.yml

# with pip
pip install -r requirements.txt

# install relso
cd ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers 

pip install -e . 
```

### Model training
```
# run training script (dataset args: gifford, GB1_WU, GFP, TAPE)
python train_relso.py  --data gifford

# run optimization algorithms
python run_optim.py --weights <path to ckpt file>/model_state.ckpt --embeddings  <path to embeddings file>train_embeddings.npy --dataset gifford
```

## Resources
1. `Transformer-based protein generation with regularized latent space optimization`: https://www.nature.com/articles/s42256-022-00532-1
2. `data`: https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers/tree/main/data
3. `code`: https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers/tree/main/relso

### Original Dataset (based on the author)
1. GIFFORD: https://github.com/gifford-lab/antibody-2019/tree/master/data/training%20data
2. GB1: https://elifesciences.org/articles/16965#data
3. GFP: https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154
4. TAPE: https://github.com/songlab-cal/tape#data
