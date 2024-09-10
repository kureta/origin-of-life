from functools import reduce
from multiprocessing import Process, shared_memory
from operator import mul
from time import sleep

import dearpygui.dearpygui as dpg
import numpy as np


def get_or_create_shared_memory(name, size):
    try:
        sm = shared_memory.SharedMemory(name=name, size=size, create=True)
    except FileExistsError:
        sm = shared_memory.SharedMemory(name=name)
        if sm.size != size:
            sm.unlink()
            get_or_create_shared_memory(name, size)

    return sm


SIZE = 32
ALIVE_PROB = 0.01
N_ITER = 128
MUTATION_RATE = 0.001
SCALE = 8
N_SPECIES = 1024


# def shuffle_batch(x):
#     return x[torch.randperm(x.shape[0], device=x.device), ...]


class App:
    def __init__(self):
        self.shape = (SIZE, SIZE, 3)
        self.size = reduce(mul, self.shape, 1) * 4  # a float32 is 4 bytes
        self.tex_buf = get_or_create_shared_memory(name="TextureBuffer", size=self.size)
        self.should_exit = get_or_create_shared_memory(name="ShouldExit", size=1)

        # NOTE: image_data will be on a buffer. it will contain 2 buffers
        #       on each state update, first pair will be written to this buffer
        #       as a single texture, reshaped to (2 * SIZE, SIZE)
        #       so there must be another `self.state` property
        #       ```
        #       img_buffer.shape = (3, 2 * SIZE, SIZE)
        #       img_buffer[...] = state_pairs[0].cpu().view(1, 2 * SIZE, SIZE)
        #       ```
        self.image_data = np.ndarray(
            self.shape, dtype=np.float32, buffer=self.tex_buf.buf
        )
        self.image_data[...] = np.zeros_like(self.image_data)

    def setup_ui(self):
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=SIZE,
                height=SIZE,
                default_value=self.image_data,  # type: ignore
                format=dpg.mvFormat_Float_rgb,
                tag="texture_tag",
            )

        with dpg.window(label="Tutorial"):
            dpg.add_image("texture_tag", width=SIZE, height=SIZE)

        dpg.set_exit_callback(self.cleanup)

    def cleanup(self):
        self.should_exit.buf[0] = 1

    def loop(self):
        import torch
        from torch.distributions import Bernoulli

        torch.autograd.set_grad_enabled(False)
        distribution = Bernoulli(ALIVE_PROB)

        tex_buf = get_or_create_shared_memory(name="TextureBuffer", size=self.size)
        image_data = np.ndarray(self.shape, dtype=np.float32, buffer=tex_buf.buf)
        while not self.should_exit.buf[0]:
            image_data[...] = (
                distribution.sample(torch.Size([SIZE, SIZE, 1])).repeat(1, 1, 3).numpy()
            )  # state[torch.randint(0, 1024, (1,))[0]].cpu().numpy()
            sleep(1)
        self.should_exit.close()
        self.tex_buf.close()

    def run(self):
        self.setup_ui()

        dpg.create_viewport(title="Origin of Life", width=800, height=600, vsync=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.toggle_viewport_fullscreen()
        dpg.set_viewport_vsync(True)

        p = Process(target=self.loop)
        p.start()

        while dpg.is_dearpygui_running():
            dpg.set_value("texture_tag", self.image_data)
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        p.join()

        self.tex_buf.unlink()
        self.should_exit.unlink()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
