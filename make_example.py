# Python program to save a
# video using OpenCV


import cv2


# Create an object to read
# from camera
input_path = 'il_1588xN.2846454247_734b.jpg'
frame = cv2.imread(input_path, cv2.IMREAD_COLOR)


# We need to set resolutions.
# so, convert them from float to integer.
frame_width = 54
frame_height = 54

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename.avi',
						cv2.VideoWriter_fourcc(*'MJPG'),
						10, size)
for _ in range(200):
	result.write(frame)

		# Display the frame
		# saved in the file
		#cv2.imshow('Frame', frame)

		# Press S on keyboard
		# to stop the process

# When everything done, release
# the video capture and video
# write objects
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
