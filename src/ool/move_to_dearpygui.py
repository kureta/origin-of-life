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


class App:
    def __init__(self):
        self.shape = (3, 100, 100)
        self.size = reduce(mul, self.shape, 1) * 4  # a float32 is 4 bytes
        self.tex_buf = get_or_create_shared_memory(name="TextureBuffer", size=self.size)
        self.should_exit = get_or_create_shared_memory(name="ShouldExit", size=1)
        self.raw_data = np.ndarray(
            self.shape, dtype=np.float32, buffer=self.tex_buf.buf
        )
        self.raw_data[...] = np.random.uniform(0, 1, self.shape).astype(np.float32)

    def setup_ui(self):
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=100,
                height=100,
                default_value=self.raw_data,
                format=dpg.mvFormat_Float_rgb,
                tag="texture_tag",
            )

        with dpg.window(label="Tutorial"):
            dpg.add_image("texture_tag")

        dpg.set_exit_callback(self.cleanup)

    def cleanup(self):
        self.should_exit.buf[0] = 1

    def loop(self):
        while not self.should_exit.buf[0]:
            self.raw_data[...] = np.random.uniform(0, 1, self.shape).astype(np.float32)
            sleep(0.2)
        self.should_exit.close()
        self.tex_buf.close()

    def run(self):
        self.setup_ui()

        dpg.create_viewport(title="Origin of Life", width=800, height=600, vsync=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.toggle_viewport_fullscreen()

        p = Process(target=self.loop)
        p.start()

        dpg.start_dearpygui()

        dpg.destroy_context()
        p.join()

        self.tex_buf.unlink()
        self.should_exit.unlink()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
