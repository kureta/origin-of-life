import torch
import torch.nn.functional as F
import pyray as pr


torch.autograd.set_grad_enabled(False)


def torus_pad(input_tensor, pad_width):
    padded_tensor = torch.cat(
        [
            input_tensor[..., -pad_width:, :],
            input_tensor,
            input_tensor[..., :pad_width, :],
        ],
        dim=-2,
    )
    padded_tensor = torch.cat(
        [
            padded_tensor[..., -pad_width:],
            padded_tensor,
            padded_tensor[..., :pad_width],
        ],
        dim=-1,
    )

    return padded_tensor


def rule(state, neighbor_count):
    return (neighbor_count == 3) | ((state == 1) & (neighbor_count == 2))


# NOTE: kernel has dimensions in_channels, out_channels, height, width
# NOTE: state has dimensions batch, in_channels, height, width
# TODO: instead of using torus_pad treat state size as one larger than the actual size
#       and identify edges with opposite inner edges
# TODO: either modify `state` property in-place or return new state. Do not mix both.

SIZE = 128


class App:
    def __init__(self) -> None:
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(800, 450, "Hello")
        self.kernel = torch.ones(1, 1, 3, 3, dtype=torch.uint8)
        self.kernel[..., 1, 1] = 0
        self.initialize_state()

    def initialize_state(self):
        self.state = torch.randint(0, 2, (1, 1, SIZE, SIZE), dtype=torch.uint8)

    def count_neighbors(self):
        state = torus_pad(self.state, 1)
        return F.conv2d(state, self.kernel)

    def update_state(self):
        neighbor_count = self.count_neighbors()
        self.state = rule(self.state, neighbor_count).to(torch.uint8)

    def update_texture(self):
        img_data = (self.state.squeeze() * 255).numpy()
        img = pr.Image(
            img_data,
            SIZE,
            SIZE,
            1,
            pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
        )
        self.tex = pr.load_texture_from_image(img)

    def update(self):
        self.update_state()
        self.update_texture()

    def loop(self):
        self.update()
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        pr.draw_texture(self.tex, 0, 0, pr.WHITE)
        pr.draw_fps(190, 200)
        pr.end_drawing()

    def start(self):
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
