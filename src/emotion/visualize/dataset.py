import matplotlib.pyplot as plt
from src.emotion.visualize.util import get_dataset_va_values


def plot_dataset():
    valence, arousal = get_dataset_va_values()
    _, ax = plt.subplots()

    ax.scatter(valence, arousal)
    ax.set_xlabel("valence")
    ax.set_ylabel("arousal")
    ax.set_title("VA values for song IDs between 1 and 2000")

    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_dataset()