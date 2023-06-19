import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # Convolutional layer 1
        modules.append(nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.Dropout2d(p=0.25))

        # Convolutional layer 2
        modules.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.Dropout2d(p=0.25))

        # Convolutional layer 3
        modules.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.Dropout2d(p=0.25))

        # Convolutional layer 4
        modules.append(nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.Dropout2d(p=0.25))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # Transposed convolutional layer 1
        modules.append(nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(256))

        # Transposed convolutional layer 2
        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(128))

        # Transposed convolutional layer 3
        modules.append(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(64))

        # Transposed convolutional layer 4
        modules.append(nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1))
        modules.append(nn.Tanh())

        self.cnn = nn.Sequential(*modules)
    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.f_mu = nn.Linear(n_features, self.z_dim)
        self.f_sig = nn.Linear(n_features, self.z_dim)
        self.f_d = nn.Linear(self.z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        f = self.features_encoder(x).view(x.shape[0], -1)
        mu, log_sigma2 = self.f_mu(f), self.f_sig(f)
        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(std)
        z = eps * std + mu
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.f_d(z).view(-1, *self.features_shape)
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            for i in range(n):
                z = torch.randn(1, self.z_dim, device=device)
                x_rec = self.decode(z)
                samples.append(x_rec.squeeze(0))
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    batch_size = x.size(0)
    input_size = x.size(1) * x.size(2) * x.size(3)
    latent_size = z_mu.size(1)
    # Compute data reconstruction loss term
    data_loss = torch.mean(torch.norm(x.view(batch_size, -1) - xr.view(batch_size, -1), dim=1) ** 2 / (x_sigma2 * input_size), dim=0)

    # Compute KL divergence loss term
    kldiv_loss = torch.mean(-1 * torch.sum(1 + z_log_sigma2 - z_mu ** 2 - z_log_sigma2.exp(), dim = 1), dim = 0)
    
    # Compute VAE loss
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
