from ool.game import Game

import pyray as pr


# NOTE: kernel has dimensions out_channels, in_channels, height, width
# NOTE: state has dimensions batch, in_channels, height, width
# TODO: either modify `state` property in-place or return new state. Do not mix both.

# TODO: list of adjustable parameters:
#       - grid size
#       - mutation rate
#       - alive cell probability during initialization
#       - number of iteration for each generation

# NOTE: make this multiprocess and vsync rendering
SIZE = 32
SCALE = 8


class App:
    def __init__(self) -> None:
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(800, 450, "Hello")
        self.tex = []
        self.toggle = False
        self.combo = pr.ffi.new("int *", 0)  # pyright: ignore[reportAttributeAccessIssue]
        self.dropping = False

        self.game = Game()

    def update_texture(self):
        self.tex.clear()
        for img_data in self.game.state[0].cpu().numpy().astype("uint8"):
            img = pr.Image(
                img_data * 255,
                SIZE,
                SIZE,
                1,
                pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
            )
            self.tex.append(pr.load_texture_from_image(img))

    def update(self):
        self.game.loop()
        self.update_texture()

    def loop(self):
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
            f"Generation: {self.game.generation}",
            600,
            10,
            20,
            pr.BLACK,
        )
        if pr.gui_dropdown_box(
            pr.Rectangle(600, 100, 100, 50), "asd;qwe;zxc", self.combo, self.dropping
        ):
            self.dropping = not self.dropping
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
