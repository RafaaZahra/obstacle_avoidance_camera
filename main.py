import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self, video_source=0, step_size=5):
        """Initialize the ObstacleDetector object."""
        self.cap = cv2.VideoCapture(video_source)  # Initialize video capture
        self.step_size = step_size  # Define step size for edge detection

    def getChunks(self, l, n):
        """Split a list into smaller chunks."""
        chunks = []
        for i in range(0, len(l), n):   
            chunks.append(l[i:i + n])
        return chunks

    def detectObstacles(self):
        """Detect obstacles in the video feed."""
        while True:
            ret, frame = self.cap.read()  # Read frame from the video capture
            img = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # Convert frame to BGRA color space
            recoverlay =  np.zeros(frame.shape, np.uint8)  # Initialize overlay for visualization

            blur = cv2.bilateralFilter(img, 9, 40, 40)  # Apply bilateral filter to reduce noise
            edges = cv2.Canny(blur, 50, 100)  # Detect edges using Canny algorithm
            img_h = img.shape[0] - 1  # Get image height
            img_w = img.shape[1] - 1  # Get image width
            EdgeArray = []  # Initialize array to store edge coordinates
            
            # Frame shape parameters
            frame_h, frame_w, frame_c = frame.shape

            for j in range(0, img_w, self.step_size):
                pixel = (j, 0)
                for i in range(img_h - 5, 0, -1):
                    if edges.item(i, j) == 255:
                        pixel = (j, i)
                        break
                EdgeArray.append(pixel)  # Store edge coordinates

            chunks = self.getChunks(EdgeArray, int(len(EdgeArray) / 3))  # Divide edge coordinates into chunks
            c = []  # Initialize array to store average coordinates of chunks

            for i in range(len(chunks) - 1):        
                x_vals = []
                y_vals = []

                for (x, y) in chunks[i]:
                    x_vals.append(x)
                    y_vals.append(y)
                
                avg_x = int(np.average(x_vals))  # Compute average x-coordinate
                avg_y = int(np.average(y_vals))  # Compute average y-coordinate
                
                c.append([avg_y, avg_x])  # Store average coordinates

                if len(c) >= 3:
                    if c[2][0] < 180:
                        print("Obstacle detect on the right")
                    elif c[0][0] < 180:
                        print("Obstacle detect on the Left")
                    elif c[1][0] < 180:
                        print("Obstacle detect on the Middle")   
                    else:
                        print("Safe Way") 

                cv2.line(recoverlay, (320, 480), (avg_x, avg_y), (255, 255, 0), 10)  # Draw line on overlay
                cv2.addWeighted(frame, 1, recoverlay, 0.25, 1, frame)  # Overlay frame and overlay

                avg_x = c[i][1]  # Get x-coordinate of obstacle
                avg_y = c[i][0]  # Get y-coordinate of obstacle
                text = f'x: {avg_x}, y: {avg_y}'  # Text to display obstacle coordinates
                cv2.putText(frame, text, (avg_x - 20, avg_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Display text on frame
                
                if len(c) >= 3:
                    intialValueLeft = c[0][0]
                    intialValueRight = c[2][0]

                    overlay = np.zeros((frame_h, frame_w,4), dtype="uint8")  # Initialize overlay with 4 channels
                    overlay[100:intialValueLeft, 102:107] = (255, 255, 0, 1) # Blue rectangle for left obstacle
                    overlay[100:intialValueRight, 522:527] = (0, 255, 0, 1) # Green rectangle for right obstacle
                    cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)  # Overlay rectangles on frame
                
                else:    
                    intialValueLeft =  150
                    intialValueRight = 150 

                cv2.imshow("frame", frame)  # Display frame

            k = cv2.waitKey(5) & 0xFF  # Wait for user input
            if k == 27:  # If Esc key is pressed
                break  # Exit the loop

        cv2.destroyAllWindows()  # Close all OpenCV windows
        self.cap.release()  # Release the video capture

if __name__ == "__main__":
    # Initialize the obstacle detector
    detector = ObstacleDetector()
    # Start detecting obstacles
    detector.detectObstacles()
