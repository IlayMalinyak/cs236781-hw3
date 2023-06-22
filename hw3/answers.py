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
        embed_dim = 64, 
        num_heads = 4,
        num_layers = 4,
        hidden_dim = 128,
        window_size = 32,
        droupout = 0.1,
        lr=5e-5,
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
while it is possible to fine tune the model using some of the internal layers, we suspect that the results would be worse compare to fine tune the last layers. generally, and similiar to CNNs, as we go deeper in the architecture (closer to the output) the learned representations are more complex and task specific. this implies that the middle layers(multihead attention bocks) capture a more general dependencies and relationships between words or tokens in the input sequence that can apply to many different tasks, while the last layers (classification head) adapts the general represantations to the specific task at hand. this analogy suggest that when we fine tune an NLP task (like sentiment analysis) we can use the general represantations as is and just fine tune the last layers. doing the opposite would not give much benefit since there's not much training needed for the middle layers (assuming that the model was pretrained on the same domain) and the classification head would not be updated. this ofcourse can change if we fine tune for a task from a different domain (general time series for example) where the middle layers dose not necesseraly represent the dependencies of our dataset. 


"""


# ==============
