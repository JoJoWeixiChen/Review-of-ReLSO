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
Main advantage: it helps create proteins with desired properties without having to explore every possible sequence.

### Architecture
<img width="760" alt="Screen Shot 2023-03-26 at 12 27 20 PM" src="https://user-images.githubusercontent.com/89158696/227793235-16453185-2940-4346-a381-f2a5fbf8bea4.png">

\
`Model input`: a set of protein sequences and their fitness measurements (latent representation)

`Model output`: the reconstructed protein sequences and the predicted fitness values for the input sequences.

\
Pseudocode:

1. Initialize ReLSO model

2. Define multitask loss function (reconstruction loss + fitness prediction loss, combining both sequence and fitness information)
<img width="747" alt="Screen Shot 2023-03-26 at 9 21 07 PM" src="https://user-images.githubusercontent.com/89158696/227825330-3d016986-bf97-446e-bc58-a0cf3ba0f19f.png">

3. For each epoch:

    a. For each batch of protein sequences (x) and fitness measurements (y) in the training dataset:
       i. Encode input sequences x into latent representations z using the encoder
       ii. Decode latent representations z into reconstructed sequences x' using the decoder
       iii. Predict fitness y' for the input sequences x using the fitness prediction head
       iv. Compute reconstruction loss (e.g., mean squared error between x and x')
       v. Compute fitness prediction loss (e.g., mean squared error between y and y')
       vi. Generate negative samples (zn) in the latent space and assign them low fitness values
       vii. Compute negative sampling loss (e.g., mean squared error between low fitness values and predictions for zn)
       viii. Compute interpolation regularization loss (e.g., differences between interpolated and true sequences)
       ix. Calculate total loss (reconstruction loss + fitness prediction loss + negative sampling loss + interpolation regularization loss)
       x. Update model parameters using the gradients of the total loss
       
Repeat steps 3a-3x for the desired number of epochs or until a stopping criterion is met



ReLSO employs a transformer-based encoder to learn the mapping from a protein sequence to its latent representation. 
While other encoding methods that rely on convolutional and recurrent architectures have demonstrated success in this domain, ReLSO chose to 
use a transformer encoding for several key reasons. 

1. The inductive bias of the transformer architecture matches the prevailing understanding that protein function is a consequence of pairwise interactions between residues (for example, catalytic triads of proteases). The transformer architecture has shown promise in several tasks relying on protein sequence for prediction. 

2. The transformer’s attention-based encoding scheme provides for interpretability by analysis of learned attention weights. 

3. Transformers have demonstrated advantages in representing long sequences, as a consequence of viewing the entire sequence during the forward pass, thus avoiding the limitations inherent to encoders based on recurrent neural networks.
