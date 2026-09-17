"""Realtime rendering of a gym environment inside a notebook.

    view = LiveView(env)            # env created with render_mode="rgb_array"
    view.show(env.render())         # once per step
    view.close()

Under the Neptune notebook frontend this uses ``neptune_nb.FrameStream``: frames go
through shared memory straight into a GPU texture, so rendering costs well under a
millisecond per step and the episode plays at the environment's own ``render_fps``.
Anywhere else (JupyterLab, VS Code) ``neptune_nb`` is either unavailable -- then this falls
back to the classic clear_output + matplotlib redraw -- or falls back by itself to an
updatable image.
"""

try:
    from neptune_nb import FrameStream
except ImportError:  # not running under Neptune and neptune_nb is not installed
    FrameStream = None


class LiveView:
    def __init__(self, env=None, fps="auto", **options):
        """fps: "auto" paces to ``env.metadata["render_fps"]`` (real time), a number sets
        the rate, None never waits. Extra options go to ``FrameStream`` (scale, filter...)."""
        if fps == "auto":
            metadata = getattr(env, "metadata", None) or {}
            fps = metadata.get("render_fps")
        self._stream = None
        if FrameStream is not None:
            spec = getattr(env, "spec", None)
            options.setdefault("title", getattr(spec, "id", None))
            self._stream = FrameStream(fps=fps, **options)

    def show(self, frame):
        if self._stream is not None:
            self._stream.show(frame)
            return
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        clear_output(wait=True)
        plt.imshow(frame)
        plt.axis("off")
        plt.show()

    def close(self):
        if self._stream is not None:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
