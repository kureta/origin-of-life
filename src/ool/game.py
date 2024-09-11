import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

torch.autograd.set_grad_enabled(False)

SIZE = 32
ALIVE_PROB = 0.07
N_ITER = 128
MUTATION_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# uint8 is faster on CPU, but not supported on GPU
DTYPE = torch.float32 if DEVICE == "cuda" else torch.uint8


def paired_klein_bottle_padding(x):
    """
    life evolves in [batch, 2, height, width] tensor
    lives are randomly paired, each channel represents a pair
    we are padding the entire batch of pairs of lives
    the padding scheme used is equivalent to joining the pairs on one edge,
    then joining the opposing edges to form a cylinder
    and then twisting/joining the top and bottom to form a klein bottle
    the twist is so that there is more surface area between the pairs
    if we don't twist each one will be joined to itself on the top and bottom
    """
    left = x[..., :, -1:].flip(1)
    right = x[..., :, :1].flip(1)
    bottom = torch.cat(
        (x[..., -1:, :1], x[..., -1:, :].flip(1, -1), x[..., -1:, -1:]), dim=3
    )
    top = torch.cat((x[..., :1, :1], x[..., :1, :].flip(1, -1), x[..., :1, -1:]), dim=3)
    x = torch.cat((left, x, right), dim=3)
    x = torch.cat((bottom, x, top), dim=2)

    return x


def paired_torus_padding(x):
    left = x[..., :, -1:].flip(1)
    right = x[..., :, :1].flip(1)
    x = torch.cat((left, x, right), dim=3)
    bottom = x[..., -1:, :]
    top = x[..., :1, :]
    x = torch.cat((bottom, x, top), dim=2)

    return x


def rule(state, neighbor_count):
    return (neighbor_count == 3) | ((state == 1) & (neighbor_count == 2))


def shuffle_batch(x):
    return x[torch.randperm(x.shape[0], device=x.device), ...]


class Game:
    # TODO: this class will be instantiated inside the multiprocessing.Process(target)
    #       everything related to pytorch and game of life will be inside this class
    #       that is to prevent CUDA problems when using multiprocessing
    #       there will be a SharedMemory that stores image data.
    #       This class will write to it and the other will read.
    #       If I can, I should trigger a memory update from raylib app
    def __init__(self) -> None:
        self.kernel = torch.ones(2, 1, 3, 3, dtype=DTYPE).to(DEVICE)
        self.kernel[..., 1, 1] = 0
        self.distribution = Bernoulli(ALIVE_PROB)
        self.state: torch.Tensor
        self.initialize_state()
        self.iter = 0
        self.generation = 0
        self.running = True

    def initialize_state(self):
        self.state = (
            self.distribution.sample(torch.Size([1024, 1, SIZE, SIZE]))
            .to(DTYPE)
            .to(DEVICE)
        )

    def shuffle_and_pair(self):
        self.state = shuffle_batch(self.state)
        self.state = self.state.view(-1, 2, SIZE, SIZE)

    def mutate(self):
        mask = torch.rand(self.state.size(), device=DEVICE) < MUTATION_RATE
        # XORing state with mask will flip the bits where mask is True
        self.state = self.state ^ mask

    def count_neighbors(self):
        state = paired_torus_padding(self.state)
        return F.conv2d(state, self.kernel, groups=2)

    def update_state(self):
        neighbor_count = self.count_neighbors()
        self.state = rule(self.state, neighbor_count).to(DTYPE)

    def loop(self):
        if self.iter % N_ITER == 0:
            self.generation += 1
            if self.state.shape[1] == 2:
                self.state = self.state.view(-1, 1, SIZE, SIZE)
                # self.mutate()
            self.shuffle_and_pair()
        self.iter += 1

        self.update_state()

    def start(self):
        while self.running:
            self.loop()


def main():
    game = Game()
    game.start()


if __name__ == "__main__":
    main()
