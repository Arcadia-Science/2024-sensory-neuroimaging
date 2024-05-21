import serial as ser
import time as tm
import argparse, sys

#define arduino port to connect to.
arduino_name ='COM4'
BUZZER_ON_TIME_S = 2
BUZZER_OFF_TIME_S = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Launch stim')
    parser.add_argument('--duration', type=int)
    parser.add_argument('--light', type=str)
    parser.add_argument('--buzzer', action='store_true')
    args = parser.parse_args()

    rec_duration_s = args.duration
    light = (args.light).upper()
    buzzer = args.buzzer

    print(rec_duration_s)
    print(light)
    print(f"buzzer: {buzzer}")

    dev=ser.Serial(arduino_name)
    if light == 'RESET':
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

        if buzzer:
            print('Starting buzzer')
            n_stims = rec_duration_s // (BUZZER_ON_TIME_S + BUZZER_OFF_TIME_S)
            print(f'Running {n_stims} buzzer stimulations')
            for i in range(n_stims):
                dev.write(b'SET STIM_STATUS 1;')
                tm.sleep(BUZZER_ON_TIME_S)
                dev.write(b'SET STIM_STATUS 0;')
                tm.sleep(BUZZER_OFF_TIME_S)

        tm.sleep(0.1)
        dev.write(b'SET ' + bytes(light, 'UTF-8') + b'_STATUS 0;')
    dev.close()
