import cv2
import numpy as np
import time

img_size = 800
scale = img_size / 20.0

def map_to_image(x,z):
    global img_size, scale
    # Map the quadcopter's position to the image coordinates
    x = int(x * scale + img_size / 2)
    z = int(img_size - z * scale - img_size / 2)
    return x, z

def draw(image,x,z,theta,u1,u2):
    global scale
    # Draw the quadcopter
    
    length = 0.2 # Length of the quadcopter's arm
    direction = np.array([np.cos(theta), np.sin(theta)])
    left_motor = np.array([x, z]) - length * direction
    right_motor = np.array([x, z]) + length * direction

    # Draw cross in the origin
    size = 0.2
    cv2.line(image, map_to_image(0,size), map_to_image(0, -size), (200, 200, 200), 2)
    cv2.line(image, map_to_image(size,0), map_to_image(-size, 0), (200, 200, 200), 2)

    # Define the thrust vectors
    left_thrust = u1*np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])*length
    right_thrust = u2*np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])*length

    # Draw the quadcopter as a line segment
    x1, z1 = map_to_image(left_motor[0], left_motor[1])
    x2, z2 = map_to_image(right_motor[0], right_motor[1])

    # Draw the action as an arrow
    cv2.arrowedLine(image, (x1, z1), (x1+int(left_thrust[0]*scale), z1-int(left_thrust[1]*scale)), (0, 0, 255), 2)
    cv2.arrowedLine(image, (x2, z2), (x2+int(right_thrust[0]*scale), z2-int(right_thrust[1]*scale)), (0, 0, 255), 2)
    cv2.line(image, (x1, z1), (x2, z2), (255, 0, 0), 2)

    return image

def nothing(x):
    pass
    
def animate(*trajectories):
    global img_size, scale
    window_name = "2D Quadcopter"        
    draw_path = False
    recording = False
    real_time = False
    t0 = 0.0
    t = 0.0
    
    # videowriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    maxT = np.max([traj['t'][-1] for traj in trajectories])
    writer = cv2.VideoWriter('output.mp4', fourcc, 100, (img_size, img_size))
    
    cv2.namedWindow(window_name)
    cv2.createTrackbar('t', window_name, 0, 1000, nothing)
    
    
    
    def get_time_from_trackbar():
        idx = cv2.getTrackbarPos('t', window_name)
        return idx * maxT / 1000.0
    
    while True:
        if recording:
            t += 0.01
        elif real_time:
            # get index based on time
            t = time.time() - t0
        else:
            # get index based on trackbar
            t = get_time_from_trackbar()
        
        # Create a white image as the background
        image = 255*np.ones((img_size, img_size, 3), dtype=np.uint8)
                    
        for traj in trajectories:
            i = max(0,np.searchsorted(traj['t'], t)-1)
            if draw_path:
                for j in range(i):
                    x1, z1 = map_to_image(traj['y'][j], traj['z'][j])
                    x2, z2 = map_to_image(traj['y'][j+1], traj['z'][j+1])
                    cv2.line(image, (x1, z1), (x2, z2), (255, 0, 0), 1)
            image = draw(
                image,
                traj['y'][i],
                traj['z'][i],
                traj['theta'][i],
                traj['ul'][i],
                traj['ur'][i]
            )
        
        key = cv2.waitKeyEx(1)

        # check if the "Esc" key is pressed
        if key == 27:  # 27 is the ASCII code for the "Esc" key
            cv2.destroyAllWindows()
            window_created = False
            break
        # record if 'r' is pressed
        if key == ord('r'):
            recording = not recording
            if recording:
                print("Recording started")
            else:
                writer.release()
                print("Recording saved in output.mp4")
        if recording:
            writer.write(image)
            # write text on the image
            cv2.putText(image, "Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # show path if 'p' is pressed
        if key == ord('p'):
            draw_path = not draw_path
        # zoom in if '1' is pressed
        if key == ord('1'):
            scale *= 1.1
        # zoom out if '2' is pressed
        if key == ord('2'):
            scale /= 1.1
        # real time if space bar is pressed
        if key == 32:
            # toggle
            real_time = not real_time
            if real_time:
                t0 = time.time()
                # start where the trackbar is
                t0 -= t
                
        # Instruction text
        cv2.putText(image, "Press 'Esc' to exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, "Press 'r' to start recording", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, "Press 'p' to show path", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, "Press 'space' for real time", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, "Press 1 or 2 to zoom in and out", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Show the image in the OpenCV window
        cv2.imshow(window_name, image)
    