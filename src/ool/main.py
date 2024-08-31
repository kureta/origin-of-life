import pyray as pr
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

torch.autograd.set_grad_enabled(False)


# swaps outer edges of 2 channels of a tensor
def swap_edges(state):
    (
        state[:, 0, -1, :],
        state[:, 0, 0, :],
        state[:, 0, :, -1],
        state[:, 0, :, 0],
        state[:, 1, -1, :],
        state[:, 1, 0, :],
        state[:, 1, :, -1],
        state[:, 1, :, 0],
    ) = (
        state[:, 1, -1, :],
        state[:, 1, 0, :],
        state[:, 1, :, -1],
        state[:, 1, :, 0],
        state[:, 0, -1, :],
        state[:, 0, 0, :],
        state[:, 0, :, -1],
        state[:, 0, :, 0],
    )

    return state


def place_glider(grid, ch, x, y):
    # Define the glider pattern in a 3x3 tensor
    glider = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32)

    _, _, height, width = grid.shape

    # Ensure coordinates are valid
    if x < 0 or y < 0 or x + 3 > width or y + 3 > height:
        raise ValueError("Glider placement is out of bounds.")

    # Place the glider in the specified location
    grid[0, ch, y : y + 3, x : x + 3] = glider

    return grid


def rule(state, neighbor_count):
    return (neighbor_count == 3) | ((state == 1) & (neighbor_count == 2))


# NOTE: kernel has dimensions out_channels, in_channels, height, width
# NOTE: state has dimensions batch, in_channels, height, width
# TODO: either modify `state` property in-place or return new state. Do not mix both.

# TODO: list of adjustable parameters:
#       - grid size
#       - mutation rate
#       - alive cell probability during initialization
SIZE = 128
ALIVE_PROB = 0.1


class App:
    def __init__(self) -> None:
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(800, 450, "Hello")
        self.kernel = torch.ones(2, 1, 3, 3, dtype=torch.uint8)
        self.kernel[..., 1, 1] = 0
        self.distribution = Bernoulli(ALIVE_PROB)
        self.state: torch.Tensor
        self.initialize_state()
        self.tex = []

    def initialize_state(self):
        # self.state = self.distribution.sample(torch.Size([1, 2, SIZE, SIZE])).to(
        #     torch.uint8
        # )
        self.state = torch.zeros(1, 2, SIZE, SIZE, dtype=torch.uint8)
        self.state = place_glider(self.state, 0, SIZE - 8, SIZE - 16)
        self.state = place_glider(self.state, 1, SIZE - 13, SIZE - 7)

    def count_neighbors(self):
        state = F.pad(self.state, (1, 1, 1, 1), mode="circular")
        # state = swap_edges(state)
        return F.conv2d(state, self.kernel, groups=2)

    def update_state(self):
        neighbor_count = self.count_neighbors()
        self.state = rule(self.state, neighbor_count).to(torch.uint8)

    def update_texture(self):
        self.tex.clear()
        for img_data in self.state.squeeze().numpy():
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
        self.update()
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        for idx, tex in enumerate(self.tex):
            pr.draw_texture(tex, SIZE * idx, 0, pr.WHITE)
        pr.draw_fps(190, 200)
        pr.end_drawing()

    def start(self):
        pr.set_target_fps(24)
        while not pr.window_should_close():
            if pr.is_key_released(pr.KeyboardKey.KEY_SPACE):
                self.initialize_state()
            self.loop()
        pr.close_window()


def main():
    app = App()
    app.start()


if __name__ == "__main__":
    main()
