import pyray as pr
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

torch.autograd.set_grad_enabled(False)


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


# NOTE: kernel has dimensions out_channels, in_channels, height, width
# NOTE: state has dimensions batch, in_channels, height, width
# TODO: either modify `state` property in-place or return new state. Do not mix both.

# TODO: list of adjustable parameters:
#       - grid size
#       - mutation rate
#       - alive cell probability during initialization
#       - number of iteration for each generation
SIZE = 32
ALIVE_PROB = 0.07
N_ITER = 128
MUTATION_RATE = 0.001
SCALE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# uint8 is faster on CPU, but not supported on GPU
DTYPE = torch.float32 if DEVICE == "cuda" else torch.uint8


def shuffle_batch(x):
    return x[torch.randperm(x.size(0), device=x.device), ...]


class App:
    def __init__(self) -> None:
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(800, 450, "Hello")
        self.kernel = torch.ones(2, 1, 3, 3, dtype=DTYPE).to(DEVICE)
        self.kernel[..., 1, 1] = 0
        self.distribution = Bernoulli(ALIVE_PROB)
        self.state: torch.Tensor
        self.initialize_state()
        self.tex = []
        self.iter = 0
        self.toggle = False
        self.generation = 0

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

    def update_texture(self):
        self.tex.clear()
        for img_data in self.state[0].cpu().numpy().astype("uint8"):
            img = pr.Image(
                img_data * 255,
                SIZE,
                SIZE,
                1,
                pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
            )
            self.tex.append(pr.load_texture_from_image(img))

    def update(self):
        self.update_state()
        self.update_texture()

    def loop(self):
        if self.iter % N_ITER == 0:
            self.generation += 1
            if self.state.shape[1] == 2:
                self.state = self.state.view(-1, SIZE, SIZE)
                # self.mutate()
            self.shuffle_and_pair()
        self.iter += 1

        self.update()
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        for idx, tex in enumerate(self.tex):
            scaled_size = SIZE * SCALE
            source = pr.Rectangle(SIZE * idx, 0, SIZE, SIZE)
            target = pr.Rectangle(scaled_size * idx, 0, scaled_size, scaled_size)
            pr.draw_texture_pro(tex, source, target, pr.Vector2(0, 0), 0, pr.WHITE)
        pr.draw_fps(600, 50)
        pr.draw_text(
            f"Generation: {self.generation}",
            600,
            10,
            20,
            pr.BLACK,
        )
        pr.end_drawing()

    def start(self):
        while not pr.window_should_close():
            if pr.is_key_released(pr.KeyboardKey.KEY_SPACE):
                self.toggle = not self.toggle
                pr.set_target_fps(8 if self.toggle else 99999)
            self.loop()
        pr.close_window()


def main():
    app = App()
    app.start()


if __name__ == "__main__":
    main()
