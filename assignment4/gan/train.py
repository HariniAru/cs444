import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(
    D,
    G,
    D_solver,
    G_solver,
    discriminator_loss,
    generator_loss,
    show_every=250,
    batch_size=128,
    noise_size=100,
    num_epochs=10,
    train_loader=None,
    device=None,
):
    """
    Train loop for GAN.
    - D, G: Discriminator and generator PyTorch models
    - D_solver, G_solver: Optimizers for D and G
    - discriminator_loss, generator_loss: Loss functions for D and G
    - show_every: How often to show generated images
    - batch_size: Batch size
    - noise_size: Dimension of the input noise vector
    - num_epochs: Number of epochs to train for
    - train_loader: DataLoader for the dataset
    - device: Device on which to train
    """

    iter_count = 0
    for epoch in range(num_epochs):
        print("EPOCH: ", (epoch + 1))
        for x, _ in train_loader:
            D.train()
            G.train()
            
            # Reset gradients
            D_solver.zero_grad()
            
            real_images = preprocess_img(x).to(device)
            logits_real = D(real_images)
            
            g_fake_seed = sample_noise(batch_size, noise_size, device=device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)
            
            # Calculate discriminator loss and take optimizer step
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()
            
            # Generator step
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size, device=device)
            fake_images = G(g_fake_seed)
            
            # Recalculate discriminator logits on fake images
            gen_logits_fake = D(fake_images)
            g_error = generator_loss(gen_logits_fake)  # calculate generator loss
            g_error.backward()
            G_solver.step()
            
            d_error = d_total_error.item()
            g_error = g_error.item()
            
            if iter_count % show_every == 0:
                print(f"Iter: {iter_count}, D: {d_error:.4f}, G:{g_error:.4f}")
                fake_images = fake_images.view(batch_size, input_channels, img_size, img_size)
                fake_images = deprocess_img(fake_images.data.cpu())
                show_images(fake_images[0:16], color=input_channels != 1)
                plt.show()
                print()
            
            iter_count += 1
