import numpy as np
import matplotlib.pyplot as plt
import pickle  # or use json if the format suits better
from datetime import datetime
import os
import time
import serial

print("Welcome to Solenoid Calibration!")
# Specify the serial port and baud rate
arduino_port = "COM3"  # Replace with your Arduino's port (e.g., "COM3" on Windows or "/dev/ttyUSB0" on Linux/Mac)
baud_rate = 115200       # Match the baud rate set in the Arduino sketch

# Open the serial connection
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on port {arduino_port} at {baud_rate} baud.")
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")

r2_threshold=0.9 #Linear fit threshold for determining good calibration
repeat = 1 # Default to query for repeat of measurements
ndt = 3 # Number of solenoid opening times to calculate for linear fit (minimum 3)
num_reps = 20 # Number of solenoid opening pulses to calculate
w = [0] * ndt  # Empty vector for storing water volumes
# Store the current date and time
current_datetime = datetime.now()
current_comp = os.getenv('COMPUTERNAME', 'unknown')  # or 'HOSTNAME' on Unix
prev_file=0
# Get the root directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the file name to load
file_name = "solenoid_calibration_results.pickle"  # Replace with your file name
# Construct the full file path
file_path = os.path.join(script_dir, file_name)
print(file_path)
calibration_data = {
    'date': None,
    'dt': [],
    'w': [],
    'dw': [],
    'b': None,
    'pc': None
}
# Prompt user to open previous calibration results file
open_previous = input("Open previous calibration results file? ([0]/[1]): ").strip()
if not open_previous or int(open_previous) == 1:
    try:
        with open(file_path, 'rb') as file:
            content = pickle.load(file)
        print(content)
        print("File content loaded successfully:")
        prev_file=1
    except FileNotFoundError:
        print(f"File '{file_name}' not found. Starting with new calibration data.")
        calibration_data = {}
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        calibration_data = {}
else:
    print("Starting with new calibration data.")

# Ask the user if they want to use previous solenoid dt values
use_previous = input("Use previous solenoid dt values? ([0]/[1]): ").strip()

if not use_previous or int(use_previous) == 1:
    try:
        # Assuming 'content' contains the previous dt values in a readable format
        dt_values = content['dt']
        print("Previous solenoid dt values loaded:",dt_values)
    except Exception as e:
        print(f"Error loading previous dt values: {e}")
        previous_dt_values = None
else:
    print("Starting with new solenoid dt values.")
    previous_dt_values = None
    # Input dialogue for entering new solenoid dt values
    dt_values = []
    for i in range(ndt):
        while True:
            try:
                dt = int(input(f"Enter solenoid opening time (ms) dt({i + 1}): "))
                dt_values.append(dt)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    print("New solenoid dt values entered:")
    print(dt_values)
#try:
initial_vol = float(input("Enter the starting volume of water (in mL): "))
print(f"Starting volume set to {initial_vol} mL.")
for j in range(len(dt_values)):
    reward_time = dt_values[j] / 1000  # dt in seconds
    print(j)
    # Clear input buffer
    while arduino.in_waiting > 4:
        arduino.readline()
    for i in range(num_reps):
        to_print = f"{dt_values[j]}:"
        print(to_print)
        arduino.write(to_print.encode())
        time.sleep(0.5)
    print('Enter cumulative water amounts (mL)')
    w[j] = float(input(f'w({j+1}): ')) - initial_vol
print(w)

#except Exception as e:
    #print(f"An error of grave nature occurred: {e}")

# Assume w, dt, nreps, and optionally params are defined earlier
dw_values = np.array(w) / num_reps * 1000  # convert to μL
b = np.polyfit(dw_values, dt_values, 1)  # fit: dt = b[0]*dw + b[1]

# Plotting
plt.figure()
plt.plot(dt_values, dw_values, '*', label='Data')
x = np.array([min(dw_values), max(dw_values)])
plt.plot(b[0]*x + b[1], x, '--', label=f't = {b[0]:.2f}*w + {b[1]:.2f}')

# Optional second plot
if prev_file:
    x2 = np.array([min(content['dw']), max(content['dw'])])
    b2 = content['b']
    plt.plot(b2[0]*x2 + b2[1], x2, 'b--', label='Previous fit')

plt.xlabel('Solenoid ON time (ms)')
plt.ylabel('Water amount (uL)')
plt.legend(loc='lower right')
plt.axis('auto')
plt.show(block=False)

# Compute and display R²
r2 = np.corrcoef(dw_values, dt_values)[0, 1] ** 2
r2 = round(r2, 2)
print(f'R² value of {r2}')

if r2 < r2_threshold:
    print('RECALIBRATE, DO NOT SAVE!!!')

# Prompt to save
s = input('Save? (0/[1]): ')
if not s or int(s) == 1:
    calibration_data['date'] = datetime.now().timetuple()
    calibration_data['dt'] = dt_values
    calibration_data['w'] = w
    calibration_data['dw'] = dw_values
    calibration_data['b'] = b
    calibration_data['pc'] = current_comp
    print(calibration_data)
    folder = script_dir  # replace this with actual folder path
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, 'solenoid_calibration_results.pickle')
    print(f'Parameters saved to {filepath}')
    # Save the calibration data to a pickle file
    output_file = os.path.join(script_dir, 'solenoid_calibration_results.pickle')
    print(output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"Calibration data saved successfully to {output_file}")
    plt.show()