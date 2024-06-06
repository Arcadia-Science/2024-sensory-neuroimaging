import time as tm

import click
import serial as ser

ARDUINO_COM = "COM1"  # arduino port (check in device manager)
STIMULATOR_ON_TIME_S = 2  # seconds that the tactile stimulator is on per stimulation
STIMULATOR_OFF_TIME_S = 4  # seconds that the tactile stimulator is off per stimulation


@click.command()
@click.option(
    "--light",
    type=str,
    required=True,
    help=(
        "Color of the LED light to turn on. Can be 'BLUE', 'VIOLET', 'ALTERNATE', or 'RESET'. "
        "If 'ALTERNATE', the blue and violet LEDs will alternate every camera frame. "
        "If 'RESET', all LEDs and stimulator will be turned off."
    ),
)
@click.option("--duration", type=int, help="Overall duration of the recording in seconds.")
@click.option("--stim", is_flag=True, help="Whether to launch the tactile stimulator.")
def run_launch_stim(duration, light, stim):
    """
    Controls the Teensy microcontroller to launch the tactile stimulator
    and the LED light for a given duration. Note: assumes Teensy is connected to COM1.

    Example usage (CLI):
        ```python launch_stim.py --duration 300 --light BLUE --stim```

        * Turns on the blue LED light for 300 seconds
        * Launches the tactile stimulator (2s on, 4s off, hardcoded above)

        ```python launch_stim.py --light RESET```

        * Turns off all LEDs and stimulator
    """

    print(f"duration (s): {duration}")
    print(f"light: {light}")
    print(f"stim: {stim}")

    dev = ser.Serial(ARDUINO_COM)
    if light == "RESET":  # turn everything off
        dev.write(b"SET " + bytes("BLUE", "UTF-8") + b"_STATUS 0;")
        tm.sleep(0.1)
        dev.write(b"SET " + bytes("VIOLET", "UTF-8") + b"_STATUS 0;")
        tm.sleep(0.1)
        dev.write(b"SET " + bytes("ALTERNATE", "UTF-8") + b"_STATUS 0;")
        tm.sleep(0.1)
        dev.write(b"SET STIM_STATUS 0;")
    else:
        dev.write(b"SET " + bytes(light, "UTF-8") + b"_STATUS 1;")
        tm.sleep(5)

        if stim:
            print("Starting tactile stimulator...")
            n_stims = duration // (STIMULATOR_ON_TIME_S + STIMULATOR_OFF_TIME_S)
            print(f"Running {n_stims} stimulations")
            for _i in range(n_stims):
                dev.write(b"SET STIM_STATUS 1;")
                tm.sleep(STIMULATOR_ON_TIME_S)
                dev.write(b"SET STIM_STATUS 0;")
                tm.sleep(STIMULATOR_OFF_TIME_S)

        tm.sleep(0.1)
        dev.write(b"SET " + bytes(light, "UTF-8") + b"_STATUS 0;")  # LED off
    dev.close()


if __name__ == "__main__":
    run_launch_stim()
