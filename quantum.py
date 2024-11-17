import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import poisson
import matplotlib.pyplot as plt


def quantum_sampling(u, x, y, nbar=10, test_num=200):
    """
    Perform quantum sampling on a given field distribution.
    """
    prob_distribution = np.abs(u)
    L = np.max(x) - np.min(x)
    N = len(x)
    X, Y = np.meshgrid(x, y)

    x0, y0 = sampling(prob_distribution, x, y, test_num)

    photon_intensity = np.zeros((N, N))
    photon_size = L / 60

    for i in range(test_num):
        n = poisson.rvs(nbar)
        photon_intensity += np.exp(
            -((X - x0[i]) ** 2 + (Y - y0[i]) ** 2) / (photon_size**2 * (n / nbar))
        )

    # plt.figure()
    # plt.imshow(photon_intensity, cmap="gray", extent=[x[0], x[-1], y[0], y[-1]])
    # plt.axis("equal")
    # plt.colorbar()
    # plt.show()

    return photon_intensity


def sampling(P, x, y, q):
    """
    Sample from a 2D probability distribution P(x, y).
    """
    P = P / np.sum(P)  # Normalize
    P_flat = P.flatten()
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Sample indices based on probability distribution
    indices = np.random.choice(len(P_flat), size=q, p=P_flat)

    x0 = X_flat[indices]
    y0 = Y_flat[indices]

    return x0, y0


def screen(u_in, x_in, y_in, lad0, z, D, pad_factor):
    """
    Simulate light propagation to a screen.
    """
    if z < 2 * D**2 / lad0:
        L = np.max(x_in) - np.min(x_in)
        N = u_in.shape[0]

        while D > (L * (1 + 2 * pad_factor)):
            pad_factor += 0.5
        scaling = 1 + 2 * pad_factor

        N_pad = int(np.ceil(pad_factor * N))
        u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")
        x_out = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        y_out = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        X_out, Y_out = np.meshgrid(x_out, y_out)

        h = (
            np.exp(1j * 2 * np.pi * z / lad0)
            / (1j * lad0 * z)
            * np.exp(1j * np.pi / lad0 / z * (X_out**2 + Y_out**2))
        )

        F_u_pad = fft2(u_pad)
        F_h = fft2(h)
        u_out = fftshift(ifft2(F_h * F_u_pad))

        mask = np.abs(x_out) < (D / 2)
        x_out = x_out[mask]
        y_out = y_out[mask]
        u_out = u_out[np.ix_(mask, mask)]
        u_out = u_out / np.max(np.abs(u_out))

    else:
        N = len(x_in)
        L = np.max(x_in) - np.min(x_in)
        k0 = 2 * np.pi / lad0

        X_in, Y_in = np.meshgrid(x_in, y_in)
        Kx = -k0 * X_in / z
        Ky = -k0 * Y_in / z
        kx = Kx[0, :]

        F_u = np.rot90(u_in, 2)
        N_pad = int(np.ceil(pad_factor * N))
        F_u_pad = np.pad(F_u, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

        dkx = kx[1] - kx[0]
        dky = dkx

        x_out = np.linspace(-np.pi / dkx, np.pi / dkx, N + 2 * N_pad)
        y_out = np.linspace(-np.pi / dky, np.pi / dky, N + 2 * N_pad)
        u_out = np.abs(ifftshift(ifft2(F_u_pad)))

    return u_out, x_out, y_out


def main():
    # Initialize parameters
    L = 10
    N = 301
    x1 = np.linspace(-L / 2, L / 2, N)
    y1 = np.linspace(-L / 2, L / 2, N)
    X1, Y1 = np.meshgrid(x1, y1)

    # Create initial field distribution
    u1 = np.where(X1**2 + Y1**2 <= 0.2, 1, 0)

    # Propagation parameters
    z1 = 25 * L
    C = 5e3
    lad0 = z1 / C
    D = 10 * L

    # Simulate propagation
    u2, x2, y2 = screen(u1, x1, y1, lad0, z1, D, 3)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(u1), extent=[x1[0], x1[-1], y1[0], y1[-1]], cmap="gray")
    plt.title("Input Field")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(u2), extent=[x2[0], x2[-1], y2[0], y2[-1]], cmap="gray")
    plt.title("Output Field")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    # Perform quantum sampling
    photon_intensity = quantum_sampling(u2, x2, y2)


if __name__ == "__main__":
    main()
