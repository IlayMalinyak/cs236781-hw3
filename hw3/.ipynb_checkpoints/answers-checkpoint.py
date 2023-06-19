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
**Your answer:**



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
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
