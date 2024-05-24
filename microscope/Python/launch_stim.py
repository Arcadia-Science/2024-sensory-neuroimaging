import argparse
import time as tm

import serial as ser

arduino_name ='COM1' # arduino port (check in device manager)
STIMULATOR_ON_TIME_S = 2 # seconds that the tactile stimulator is on per stimulation
STIMULATOR_OFF_TIME_S = 4 # seconds that the tactile stimulator is off per stimulation

if __name__ == '__main__':
    '''
    Controls the Teensy microcontroller to launch the tactile stimulator
    and the LED light for a given duration. Note: assumes Teensy is connected to COM1.

    Args:
        --duration: int, overall duration of the recording in seconds
        --light: str, color of the LED light to turn on. Can be 'BLUE', 'VIOLET', 'ALTERNATE',
                or 'RESET'. If 'ALTERNATE', the blue and violet LEDs will alternate every camera frame.
                If 'RESET', all LEDs and stimulator will be turned off.
        --stim: bool, whether to launch the tactile stimulator

    Example usage (CLI):
        ```python launch_stim.py --duration 300 --light BLUE --stim```

        * Turns on the blue LED light for 300 seconds
        * Launches the tactile stimulator (2s on, 4s off, hardcoded above)

        ```python launch_stim.py --light RESET```

        * Turns off all LEDs and stimulator
    '''
    parser = argparse.ArgumentParser(
                    prog='Launch stim')
    parser.add_argument('--duration', type=int)
    parser.add_argument('--light', type=str)
    parser.add_argument('--stim', action='store_true')
    args = parser.parse_args()

    rec_duration_s = args.duration
    light = (args.light).upper()
    stim = args.stim

    print(rec_duration_s)
    print(light)
    print(f"stim: {stim}")

    dev=ser.Serial(arduino_name)
    if light == 'RESET': # turn everything off
        dev.write(b'SET ' + bytes('BLUE', 'UTF-8') +b'_STATUS 0;')
        tm.sleep(0.1)
        dev.write(b'SET ' + bytes('VIOLET', 'UTF-8') +b'_STATUS 0;')
        tm.sleep(0.1)
        dev.write(b'SET ' + bytes('ALTERNATE', 'UTF-8') +b'_STATUS 0;')
        tm.sleep(0.1)
        dev.write(b'SET STIM_STATUS 0;')
    else:
        dev.write(b'SET ' + bytes(light, 'UTF-8') +b'_STATUS 1;')
        tm.sleep(5)

        if stim:
            print('Starting tactile stimulator...')
            n_stims = rec_duration_s // (STIMULATOR_ON_TIME_S + STIMULATOR_OFF_TIME_S)
            print(f'Running {n_stims} stimulations')
            for _i in range(n_stims):
                dev.write(b'SET STIM_STATUS 1;')
                tm.sleep(STIMULATOR_ON_TIME_S)
                dev.write(b'SET STIM_STATUS 0;')
                tm.sleep(STIMULATOR_OFF_TIME_S)

        tm.sleep(0.1)
        dev.write(b'SET ' + bytes(light, 'UTF-8') + b'_STATUS 0;') # LED off
    dev.close()
