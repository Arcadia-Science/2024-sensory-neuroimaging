import nidaqmx
from nidaqmx.system import System
import numpy as np
import time

""" Code written with Chat-GPT4 to control a National instruments DAQ USB-6001
for use with neuroimaging. MR 2024-02-13"""

def check_device_availability(device_name="Dev1"):
    """
    Checks if the specified device is available in the system.
    Prints a message indicating whether the device is found and if it's simulated.
    """
    system = System.local()
    if device_name in system.devices:
        device = system.devices[device_name]
        print(f"Device {device_name} found. Simulated: {device.is_simulated}.")
    else:
        print(f"Device {device_name} not found. Please ensure the device is connected and configured correctly.")


def read_inputs(task):
    # Read the values from the three analog inputs
    data = task.read(number_of_samples_per_channel=1)
    return data


def generate_square_wave(frequency, duration, sample_rate=1000, delay=0):
    # Generate a square wave with a given frequency and duration, including a delay represented by zeroes
    # Calculate the number of samples needed for the delay
    delay_samples = int(sample_rate * delay)
    # Generate the square wave without delay
    t = np.arange(0, duration, 1 / sample_rate)
    wave = 0.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
    # Prepend zeroes to the wave to simulate the delay
    delayed_wave = np.concatenate((np.zeros(delay_samples), wave))
    return delayed_wave


def main(device_name ="Dev1", sample_rate=10000, output_delay=20, output_duration=2, output_frequency=0.25):
    check_device_availability(device_name)  # Specify your device name here

    try:
        with nidaqmx.Task() as input_task, nidaqmx.Task() as output_task:
            input_task.ai_channels.add_ai_voltage_chan(
                "Dev1/ai0:2"
            )  # Adjust the channel names as needed

            output_task.ao_channels.add_ao_voltage_chan(
                "Dev1/ao0"
            )  # Adjust the channel name as needed
  
            output_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            )

            square_wave = generate_square_wave(
                output_frequency, output_duration, sample_rate, output_delay
            )
            output_task.write(square_wave, auto_start=True)

            # Record
            print("Reading inputs...")
            data = read_inputs(input_task)
            print("Input Data:", data)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
