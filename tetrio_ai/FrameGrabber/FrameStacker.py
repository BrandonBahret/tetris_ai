import math
import types


class FrameStacker:
    def __init__(self, callback, ignore_period, stack_length):
        isfunction = isinstance(callback, types.FunctionType)
        ismethod = isinstance(callback, types.MethodType)
        if not isfunction | ismethod:
            raise TypeError(f"'{callback}' is not a function.")

        self.callback = callback
        self.pulse_gap = ignore_period
        self.pulse_width = stack_length

        self.frames = []
        self.frame_counter = 0

    def square_wave(self, x, pulse_gap=1, pulse_width=1):
        if pulse_gap == 0:
            return pulse_width != 0

        cycle = pulse_width + pulse_gap
        x = x % cycle

        n = (x % pulse_gap) + 1
        d = x + 1
        quotient = n / d
        
        y = -math.floor(quotient) +1
        return y > 0

    def should_accumulate(self):
        x = self.frame_counter
        return self.square_wave(x, self.pulse_gap, self.pulse_width)

    def should_return_stack(self):
        x = self.frame_counter
        offset_pulse_gap = self.pulse_gap + self.pulse_width - 1
        return self.square_wave(x, offset_pulse_gap, 1)

    def stack_frame(self, frame):
        # callback = self.callback

        if self.frame_counter == 0:
            # PerformanceChecker.check_performance(callback, [frame])
            self.callback([frame])

        if self.should_accumulate():
            self.frames.append(frame)
        
        if self.should_return_stack():
            # PerformanceChecker.check_performance(callback, self.frames)
            self.callback(self.frames)
            self.frames.clear()

        self.frame_counter += 1
