import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def point_creator(x0, y0, x, y):
    X, Y = np.meshgrid(x, y)
    out = np.zeros((len(y), len(x)))
    out[(X == x0) & (Y == y0)] = 1
    return out


def free_space(lad0, z, u_in, x, y, pad_factor):
    Lx = np.max(x) - np.min(x)
    Ly = np.max(y) - np.min(y)
    y_num, x_num = u_in.shape
    scaling = 1 + 2 * pad_factor
    N_pad = int(pad_factor * max(x_num, y_num))

    # Pad input array
    u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

    x_new = np.linspace(-Lx / 2 * scaling, Lx / 2 * scaling, x_num + 2 * N_pad)
    y_new = np.linspace(-Ly / 2 * scaling, Ly / 2 * scaling, y_num + 2 * N_pad)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    h = (
        np.exp(1j * 2 * np.pi * z / lad0)
        / (1j * lad0 * z)
        * np.exp(1j * np.pi / (lad0 * z) * (X_new**2 + Y_new**2))
    )

    F_u_pad = fft2(u_pad)
    F_h = fft2(h)

    out = fftshift(ifft2(F_h * F_u_pad))
    out = out / np.max(np.abs(out))

    return out, x_new, y_new


def ring_apperture(u_in, x_in, y_in, R, lad0, z, pad_factor):
    if z < 2 * (2 * R) ** 2 / lad0:
        L = np.max(x_in) - np.min(x_in)
        N = u_in.shape[0]

        while (2 * R) > (L * (1 + 2 * pad_factor)):
            pad_factor += 0.5

        scaling = 1 + 2 * pad_factor
        N_pad = int(pad_factor * N)

        u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")
        x2 = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        y2 = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        X2, Y2 = np.meshgrid(x2, y2)

        h = (
            np.exp(1j * 2 * np.pi * z / lad0)
            / (1j * lad0 * z)
            * np.exp(1j * np.pi / (lad0 * z) * (X2**2 + Y2**2))
        )

        F_u_pad = fft2(u_pad)
        F_h = fft2(h)

        u2 = fftshift(ifft2(F_h * F_u_pad))
        u2 = u2 / np.max(np.abs(u2))

        # Circular aperture function
        t_aperture = np.where(np.sqrt(X2**2 + Y2**2) <= R, 1, 0)
        u_out = u2 * t_aperture

        I = np.where(np.abs(x2) < R)[0]
        x_out = x2[I]
        y_out = y2[I]
        u_out = u_out[np.ix_(I, I)]

    else:
        N = len(x_in)
        k0 = 2 * np.pi / lad0
        X_in, Y_in = np.meshgrid(x_in, y_in)

        Kx = -k0 * X_in / z
        Ky = -k0 * Y_in / z
        kx = Kx[0, :]

        F_u = np.rot90(u_in, 2)
        N_pad = int(pad_factor * N)
        F_u_pad = np.pad(F_u, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

        dkx = kx[1] - kx[0]
        dky = dkx

        x_out = np.linspace(-np.pi / dkx, np.pi / dkx, N + 2 * N_pad)
        y_out = np.linspace(-np.pi / dky, np.pi / dky, N + 2 * N_pad)

        u_out = np.abs(ifftshift(ifft2(F_u_pad)))

    return u_out, x_out, y_out


def ring_lens(u_in, x_in, y_in, R, lad0, f, z, pad_factor):
    if z < 2 * (2 * R) ** 2 / lad0:
        L = np.max(x_in) - np.min(x_in)
        N = u_in.shape[0]

        while (2 * R) > (L * (1 + 2 * pad_factor)):
            pad_factor += 0.5

        scaling = 1 + 2 * pad_factor
        N_pad = int(pad_factor * N)

        u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")
        x2 = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        y2 = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        X2, Y2 = np.meshgrid(x2, y2)

        h = (
            np.exp(1j * 2 * np.pi * z / lad0)
            / (1j * lad0 * z)
            * np.exp(1j * np.pi / (lad0 * z) * (X2**2 + Y2**2))
        )

        F_u_pad = fft2(u_pad)
        F_h = fft2(h)

        u2 = fftshift(ifft2(F_h * F_u_pad))

        # Lens transmission function
        r2 = X2**2 + Y2**2
        t_lens = np.where(np.sqrt(r2) <= R, np.exp(-1j * np.pi / (lad0 * f) * r2), 0)
        u_out = u2 * t_lens

        I = np.where(np.abs(x2) < R)[0]
        x_out = x2[I]
        y_out = y2[I]
        u_out = u_out[np.ix_(I, I)]

    else:
        N = len(x_in)
        k0 = 2 * np.pi / lad0
        X_in, Y_in = np.meshgrid(x_in, y_in)

        Kx = -k0 * X_in / z
        Ky = -k0 * Y_in / z
        kx = Kx[0, :]

        F_u = np.rot90(u_in, 2)
        N_pad = int(pad_factor * N)
        F_u_pad = np.pad(F_u, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

        dkx = kx[1] - kx[0]
        dky = dkx

        x_out = np.linspace(-np.pi / dkx, np.pi / dkx, N + 2 * N_pad)
        y_out = np.linspace(-np.pi / dky, np.pi / dky, N + 2 * N_pad)

        u_out = np.abs(ifftshift(ifft2(F_u_pad)))

    return u_out, x_out, y_out


def screen(u_in, x_in, y_in, lad0, z, D, pad_factor):
    if z < 2 * D**2 / lad0:
        L = np.max(x_in) - np.min(x_in)
        N = u_in.shape[0]

        while D > (L * (1 + 2 * pad_factor)):
            pad_factor += 0.5

        scaling = 1 + 2 * pad_factor
        N_pad = int(pad_factor * N)

        u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")
        x_out = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        y_out = np.linspace(-L / 2 * scaling, L / 2 * scaling, N + 2 * N_pad)
        X_out, Y_out = np.meshgrid(x_out, y_out)

        h = (
            np.exp(1j * 2 * np.pi * z / lad0)
            / (1j * lad0 * z)
            * np.exp(1j * np.pi / (lad0 * z) * (X_out**2 + Y_out**2))
        )

        F_u_pad = fft2(u_pad)
        F_h = fft2(h)

        u_out = fftshift(ifft2(F_h * F_u_pad))

        I = np.where(np.abs(x_out) < (D / 2))[0]
        x_out = x_out[I]
        y_out = y_out[I]
        u_out = u_out[np.ix_(I, I)]

    else:
        N = len(x_in)
        k0 = 2 * np.pi / lad0
        X_in, Y_in = np.meshgrid(x_in, y_in)

        Kx = -k0 * X_in / z
        Ky = -k0 * Y_in / z
        kx = Kx[0, :]

        F_u = np.rot90(u_in, 2)
        N_pad = int(pad_factor * N)
        F_u_pad = np.pad(F_u, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

        dkx = kx[1] - kx[0]
        dky = dkx

        x_out = np.linspace(-np.pi / dkx, np.pi / dkx, N + 2 * N_pad)
        y_out = np.linspace(-np.pi / dky, np.pi / dky, N + 2 * N_pad)

        u_out = np.abs(ifftshift(ifft2(F_u_pad)))

    return u_out, x_out, y_out


# Example usage for single slit simulation
def simulate_single_slit():
    L = 10
    N = 201
    x1 = np.linspace(-L / 2, L / 2, N)
    y1 = np.linspace(-L / 2, L / 2, N)
    X1, Y1 = np.meshgrid(x1, y1)

    # Rectangle function
    def func_rect(x, y):
        return np.where((np.abs(x) <= 2) & (np.abs(y) <= 0.3), 1, 0)

    u1 = func_rect(X1, 0.75 * Y1)

    z1 = 25 * L
    C = 5e3
    lad0 = z1 / C
    D = 10 * L

    u2, x2, y2 = screen(u1, x1, y1, lad0, z1, D, 3)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.abs(u1), cmap="gray", extent=[x1[0], x1[-1], y1[0], y1[-1]])
    plt.title("Input Field")
    plt.axis("equal")

    plt.subplot(122)
    plt.imshow(np.abs(u2), cmap="gray", extent=[x2[0], x2[-1], y2[0], y2[-1]])
    plt.title("Output Field")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    simulate_single_slit()
