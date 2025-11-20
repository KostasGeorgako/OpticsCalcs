import cv2
import numpy as np

# --- Mouse callback ---
def draw_boundaries(event, x, y, flags, state):

    if state["drawing_mode"] == "default":
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state["dragging"] = True
        state["points"] = [(x,y), (x,y)]

    elif event == cv2.EVENT_MOUSEMOVE and state.get("dragging"):
        state["points"][1] = (x, y)


    elif event == cv2.EVENT_LBUTTONUP and len(state.get("points")) > 0:
        state.update(dragging=False)
        state["points"][1] = (x, y)
        state["loaded_points"] = True
        state["dragging"] = False

        print("Loaded points")


def reset_state_and_change_mode(state, mode):
    state["points"] = []
    state["loaded_points"] = False
    state["dragging"] = False

    print(f"Changed state {state["drawing_mode"]} -> {mode}")
    state["drawing_mode"] = mode

    if mode == "default":
        state["avg_hsv_roi"] = None


cv2.namedWindow("Output")

cap = cv2.VideoCapture(0)

state = {
    "points": [],
    "dragging": False,
    "drawing_mode": "default",
    "loaded_points": False,
    "avg_hsv_roi": None
}

# h_tol, s_tol, v_tol = 5, 60, 30
h_tol, s_tol, v_tol = 10, 40, 40

cv2.setMouseCallback("Output", draw_boundaries, state)

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)

    if state["drawing_mode"] == "hsv_roi" and state["loaded_points"]:
        p1, p2 = state["points"]
        x1, y1, x2, y2 = min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])

        hsv_roi = hsv_frame[y1:y2, x1:x2]
        print(hsv_roi)
        # h, s, v (without avg alpha)
        state["avg_hsv_roi"] = cv2.mean(hsv_roi)[:3]

        # forward the state after selection
        reset_state_and_change_mode(state, "boundaries")


    masked_frame = None

    if state["avg_hsv_roi"] != None:
        h_avg, s_avg, v_avg = state["avg_hsv_roi"]
        lower = np.array([max(h_avg - h_tol, 0),
                          max(s_avg - s_tol, 0),
                          max(v_avg - v_tol, 0)], dtype=np.uint8)
        upper = np.array([min(h_avg + h_tol, 179),
                          min(s_avg + v_tol, 255),
                          min(v_avg + v_tol, 255)], dtype=np.uint8)

        masked_frame = cv2.inRange(hsv_frame, lower, upper)
    else:
        masked_frame = cv2.inRange(hsv_frame, np.array([0, 0, 0]), np.array([179, 255, 255]))

    drawed_frame = cv2.bitwise_and(curr_frame, curr_frame, mask=masked_frame)


    if len(state["points"]) > 0:

        colour = (0, 0, 255) if state["dragging"] else (0, 255, 0)

        p1 = state["points"][0]
        p2 = state["points"][1]
        cv2.rectangle(drawed_frame, p1, p2, colour, 1)

    cv2.imshow("Output", drawed_frame)



    key = cv2.waitKey(1) & 0xFF

    # reset key
    if key == ord('b'):
        reset_state_and_change_mode(state, "boundaries")

    elif key == ord('h'):
        reset_state_and_change_mode(state, "hsv_roi")

    elif key == ord('r'):
        reset_state_and_change_mode(state, "default")


    # ESC
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
