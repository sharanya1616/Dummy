import serial
import time

# Open serial port (check your port and baud rate)
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Allow some time for Arduino to initialize

# Send trigger variable
ser.write(b'1')  # Send '1' to move servo
time.sleep(6)    # Wait for 6 seconds (5 seconds for movement + 1 second buffer)

# Close serial port
ser.close()
