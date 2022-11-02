import os


files = os.listdir("/media/pranoy/Pranoy/human3.6M/images/")
for path in files:
	
	file = os.system(f"ls /media/pranoy/Pranoy/human3.6M/images/{path} >> z.txt")
