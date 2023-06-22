r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers   



def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

"""

part1_q2 = r"""
**Your answer:**

"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=32, h_dim=512, z_dim=2, x_sigma2=3.7, learn_rate=0.00033, betas=(0.96, 0.99),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
the role of $\sigma^2$ can be understood from the two terms it appears in. 
<br>
first, the likelihood equation: $p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$ from this we can see that $\sigma^2$ control the variance of the likelihood distribution around the decoder output. this correspond to variance in the generated instances given a latent representation. low $\sigma^2$ leads to generated instances be more concentrated and tightly clustered around the point calculated by the decoder. this results in high fidelity to the training data (overfitting). from the other hand, it might not be able to generalize instances (different pose for example) well. high $\sigma^2$ leads to the opposite - more diversity in instances but can also introduce more randomness and noise into the generated instances, potentially deviating from the training data.
<br>
the second term that includes $\sigma^2$ is the loss. it appears in the denominator of the reconstruction error and serve as a parameter that balances the two terms of the loss. high $\sigma^2$ gives more weight to the KL term which has the role of regularisation term - increases the variance and diversity. low $\sigma$ gives more weight to the reconstruction term which fit the instances directly to the training data.
<br> we see that the role of $\sigma^2$ coincides - control the amount of regularization and variance of sampled instances 


"""

part2_q2 = r"""
1. The reconstruction loss measures how well the model is able to reconstruct the input data from the latent space. It encourages the decoder network to generate outputs that are similar to the original input data.
<br>
The regularization loss encourages the encoder to learn a regularized latent space. Its goal is to ensure that the latent space follows a specific probability distribution (normal in our case), and in doing so it ensures diveresity, continuity and meaningfulness of the instances. 
<br>
2. the KL term regularize the latent space by enforcing it to be of specific distribution. Without a well defined regularisation term, the model can learn, in order to minimise its reconstruction error, to “ignore” the fact that distributions are returned and behave almost like classic autoencoders (leading to overfitting). To do so, the encoder can either return distributions with tiny variances (that would tend to be punctual distributions) or return distributions with very different means. In order to avoid these effects we have to regularise both the covariance matrix and the mean of the distributions returned by the encoder, which done by the KL loss term. this would increase the continuity (close points in the latent space become close decoded samples) and the completenes (points sampled from the latent space are meaninfull) of the latent space.
<br>
the benefits are:
<br>
Regularization: By aligning the approximate posterior distribution with the prior distribution, the KL loss acts as a regularization term. It helps prevent overfitting and increase the variability of the model.
well behaved and Controllable latent space: as mentioned above, the KL term induce continuity and smoothness of the latent space. since we can describe the latent space by two parameters (std and mean) it is also controllable.    


"""

part2_q3 = r"""
The evidence distribution refers to the distribution of the input data given the latent space representation. It represents the likelihood of generating the input data from a sample in the latent space. Maximizing the evidence distribution ensures that the reconstructed data resembles the original data as closely as possible. By maximizing the evidence distribution, we encourage the decoder to generate reconstructions that capture the important features and characteristics of the input data. This promotes the learning of meaningful representations in the latent space and improves the overall quality of the generated samples.
<br> 
as explained in the lecture, the evidence distribution itself is intractable but we can get a lower bound on it



"""

part2_q4 = r"""
in a VAE we sample $z$ from a normal distribution. modeling the log of its variance instead of the actual variance has some benefits: 
<br>
1.it is more numericaly stable to work in log space, espacially when the variance is small.
<br>
2. its aligned with the assumption that the distribution is normal with diagonal covariance matrix -  for multivariate Gaussian with a diagonal covariance matrix, the matrix is exponentiated elementwise so by taking the log of the matrix, we get the actual variance elements. this is also the motivation to take $log(sigma^2)$ as explained in the VAE paper.  



"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 128, 
        num_heads = 4,
        num_layers = 2,
        hidden_dim = 192,
        window_size = 128,
        droupout = 0.22,
        lr=0.000549,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
when stacking multiple layers in a CNN architecture, each layer gets an input which was proccessed using a recpetive field defined by the previous layer's kernel and proccess it using the current layer's receptive field. thus, each layer increase the effective total receptive field of the input with respect to the final layer. In a sliding window there is a similiar effect - each layer processes the input sequence with a sliding window mechanism using an attention mechanism that attends to a fixed-sized window of neighboring positions and capture the dependencies and interactions within that window. By stacking multiple layers, each layer receives input from a larger context captured by the previous layer and increase the effective total context window at the final layer.  

"""

part3_q2 = r"""
one possible option that was suggested in the paper is the following - combining global attention for a small number of tokens together with sliding window. The tokens that uses the golbal attention attend to all other tokens and all other tokens asttend to them so they can proccess the globl information, while the tokens that uses sliding window attention proccess local context. we can spread the global attention tokens evenly accross the sentence. in a classification task it is recommended that the classification token (CLS in our case) would use global attenion. 


"""


part4_q1 = r"""
comparing to part3, fine tuning outperfomrs training from scratch. this fact comes from several reasons - the model we fine tuned is much larger than the model we built from scratch so it has more expressivness. in addition, it was trained on a much larger datasets in the pretrained phase. in addition, the tasks that the pretrained model was trained on in the pretraining phase are similiar to the downstream task. this makes the context that was learned in the  pretraining phase "relevant" for the downstream task and leads to better results (using only training of the classifiers) than training from scratch. when using a pretrained model from the same domain and on tasks that are aimiliar to the pretrianing, it is usually the case that fine tuning would give better results than trianing from scratch (we saw it also in vision models, that contrastive learning outperfomrs training from scratch). if the task is very specific and different in nature, or if the domain is different (for example, using BERT for non NLP time series classification) we might encounter different results and find that training from scratch is better.  


"""

part4_q2 = r"""
while it is possible to fine tune the model using some of the internal layers, we suspect that the model would be able to learn but the results would be worse compare to fine tuning the last layers. generally, and similiar to CNNs, as we go deeper in the architecture (closer to the output) the learned representations are more complex and task specific. this implies that the middle layers(multihead attention bocks) capture a more general dependencies and relationships between words or tokens in the input sequence that can apply to many different tasks, while the last layers (classification head) adapts the general represantations to the specific task at hand. this analogy suggest that when we fine tune an NLP task (like sentiment analysis) we can use the general represantations as is and just fine tune the last layers. doing the opposite can lead to improvement (by improving the general represantations and adapt them to the specific task) but would not give much benefit since there's not much training needed for the middle layers (assuming that the model was pretrained on the same domain) and the classification head would not be updated. this ofcourse can change if we fine tune for a task from a different domain (general time series for example) where the middle layers dose not necesseraly represent the dependencies of our dataset. 


"""


# ==============
