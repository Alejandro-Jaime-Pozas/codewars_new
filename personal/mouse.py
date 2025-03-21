import pyautogui
import time

# Function to switch to the next window
def switch_window():
    pyautogui.keyDown('alt')  # Hold down the 'alt' key
    pyautogui.press('tab')    # Press 'tab' to switch to the next window
    pyautogui.keyUp('alt')    # Release the 'alt' key
    
# Loop to continuously switch windows
while True:
    switch_window()          # Switch to the next window
    time.sleep(15)            # Wait for 2 seconds before switching again



# import pyautogui
# import random
# import time

# try:
#     while True:
#         x, y = random.randint(100, 500), random.randint(100, 500)
#         duration = random.uniform(0.1, 0.5)
#         pyautogui.moveTo(x, y, duration=duration)
#         time.sleep(random.uniform(0.1, 0.6))

# except KeyboardInterrupt:
#     print('Script interrupted by')



# import pyautogui
# import time

# # Define the interval (in seconds)
# interval = 2

# try:
#     count = 0
#     n = 0
#     while True:
#         # Get the current mouse position
#         x, y = pyautogui.position()
        
#         # Move the mouse cursor down by 1 pixel
#         if n == 0:
#             pyautogui.moveTo(x, y + 250)
#             n = 1
#         elif n == 1:
#             pyautogui.moveTo(x, y + -250)
#             n = 0

        
#         # Wait for the specified interval
#         time.sleep(interval)

#         count += 1

# except KeyboardInterrupt:
#     print("Script stopped by user.")
