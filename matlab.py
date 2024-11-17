import numpy as np
import matplotlib.pyplot as plt
import cv2


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

    # Pad the input array
    u_pad = np.pad(u_in, ((N_pad, N_pad), (N_pad, N_pad)), mode="constant")

    # Create new coordinate grids
    x_new = np.linspace(-Lx / 2 * scaling, Lx / 2 * scaling, x_num + 2 * N_pad)
    y_new = np.linspace(-Ly / 2 * scaling, Ly / 2 * scaling, y_num + 2 * N_pad)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    # Create transfer function
    h = (
        np.exp(1j * 2 * np.pi * z / lad0)
        / (1j * lad0 * z)
        * np.exp(1j * np.pi / (lad0 * z) * (X_new**2 + Y_new**2))
    )

    # Perform convolution in Fourier domain
    F_u_pad = np.fft.fft2(u_pad)
    F_h = np.fft.fft2(h)

    # Calculate output field
    out = np.fft.fftshift(np.fft.ifft2(F_h * F_u_pad))
    out = out / np.max(np.abs(out))

    return out, x_new, y_new


def ring_aperture(u_in, x0, y0, R, x, y):
    X, Y = np.meshgrid(x, y)
    aperture = np.where(np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) <= R, 1, 0)
    u_out = u_in * aperture

    limit = (max(abs(x0), abs(y0)) + R) * 1.5
    x_mask = np.abs(x) <= limit
    y_mask = np.abs(y) <= limit
    x_new = x[x_mask]
    y_new = y[y_mask]
    u_out = u_out[
        np.min(np.where(x_mask)[0]) : np.max(np.where(x_mask)[0]) + 1,
        np.min(np.where(x_mask)[0]) : np.max(np.where(x_mask)[0]) + 1,
    ]

    return u_out, x_new, y_new


def ring_lens(u_in, x0, y0, R, lad0, f, x, y):
    X, Y = np.meshgrid(x, y)
    phase = np.exp(-1j * np.pi / (lad0 * f) * ((X - x0) ** 2 + (Y - y0) ** 2))
    aperture = np.where(np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) <= R, 1, 0)
    u_out = u_in * phase * aperture

    limit = (max(abs(x0), abs(y0)) + R) * 1.5
    x_mask = np.abs(x) <= limit
    y_mask = np.abs(y) <= limit
    x_new = x[x_mask]
    y_new = y[y_mask]
    u_out = u_out[
        np.min(np.where(x_mask)[0]) : np.max(np.where(x_mask)[0]) + 1,
        np.min(np.where(x_mask)[0]) : np.max(np.where(x_mask)[0]) + 1,
    ]

    return u_out, x_new, y_new


def high_pass(u_in, x, y, factor):
    X, Y = np.meshgrid(x, y)
    R = factor * (np.max(x) - np.min(x)) / 2
    filter_mask = (X**2 + Y**2) > R**2
    return np.fft.ifft2(filter_mask * np.fft.fftshift(np.fft.fft2(u_in)))


# Main simulation
def main():
    # Define rectangular function
    def func_rect(x, y):
        return np.where((np.abs(x) <= 2) & (np.abs(y) <= 1), 1, 0)

    # Initialize parameters
    L = 10
    N = 701
    x1 = np.linspace(-L / 2, L / 2, N)
    y1 = np.linspace(-L / 2, L / 2, N)
    X1, Y1 = np.meshgrid(x1, y1)

    # Create initial field
    u1 = func_rect(X1, Y1)
    lad0 = L / 1000

    # Propagation parameters
    z1 = 10 * L
    f = 0.5 * z1
    z2 = z1 * f / (z1 - f)

    # Perform propagation
    u2, x2, y2 = free_space(lad0, z1, u1, x1, y1, 1)
    u3, x3, y3 = ring_lens(u2, 0, 0, 10 * L, lad0, f, x2, y2)
    u4, x4, y4 = free_space(lad0, z2, u3, x3, y3, 1)

    # Plot results
    fig = plt.figure(figsize=(12, 10))

    # Initial field
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(np.abs(u1), cmap="bone")
    ax1.set_title("Initial Field")

    # After first propagation
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(np.abs(u2), cmap="bone")
    ax2.set_title("After First Propagation")

    # After lens
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(np.abs(u3), cmap="bone")
    ax3.set_title("After Lens")

    # Final field
    mask_x = np.abs(x4) <= L / 2
    mask_y = np.abs(y4) <= L / 2
    u4_cropped = u4[np.ix_(mask_y, mask_x)]
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(np.abs(u4_cropped), cmap="bone")
    ax4.set_title("Final Field")

    plt.tight_layout()
    plt.show()

    # High-pass filtering visualization
    if True:
        fig = plt.figure(figsize=(12, 10))

        ui = u4_cropped

        # Original field
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(np.abs(ui), cmap="bone")
        ax1.set_title("Original Field")

        # Fourier transform
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(np.abs(np.fft.fftshift(np.fft.fft2(ui))), cmap="bone")
        ax2.set_title("Fourier Transform")

        # High-pass filtered
        uu = high_pass(ui, x1, y1, 0.3)
        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(np.abs(uu), cmap="bone")
        ax3.set_title("High-pass Filtered")

        # Filtered Fourier transform
        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(np.abs(np.fft.fft2(uu)), cmap="bone")
        ax4.set_title("Filtered Fourier Transform")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    l = 10
    n = 201
    x1 = np.linspace(-l // 2, l // 2, n)
    y1 = np.linspace(-l // 2, l // 2, n)
    lad0 = l / 1000
    z1 = 10 * l
    f = 0.5 * z1
    d = 10 * l
    pad_factor = 1
    z2 = z1 * f / (z1 - f)

    u1 = cv2.imread("abc.png", cv2.IMREAD_GRAYSCALE)
    # print(u_in.shape)

    u2, x2, y2 = free_space(lad0, z1, u1, x1, y1, pad_factor=pad_factor)

    u3, x3, y3 = ring_lens(u2, 0, 0, d / 2, lad0, f, x2, y2)
    u4, x4, y4 = free_space(lad0, z2, u3, x3, y3, pad_factor=pad_factor)

    # Plot results
    fig = plt.figure(figsize=(12, 10))

    # Initial field
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(np.abs(u1), cmap="gray")
    ax1.set_title("Initial Field")

    # After first propagation
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(np.abs(u2), cmap="gray")
    ax2.set_title("After First Propagation")

    # After lens
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(np.abs(u3), cmap="gray")
    ax3.set_title("After Lens")

    # Final field
    mask_x = np.abs(x4) <= l / 2
    mask_y = np.abs(y4) <= l / 2
    u4_cropped = u4[np.ix_(mask_y, mask_x)]
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(np.abs(u4_cropped), cmap="gray")
    ax4.set_title("Final Field")

    plt.tight_layout()
    plt.show()
