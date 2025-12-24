from typing import Tuple

import numpy as np
import cv2
import time
from operator import itemgetter

import sounddevice as sd
from scipy import signal
import asyncio


class DynamicLaserTracker:
    def __init__(self):
        self.is_calibrated = False
        self.center_found = False

        self.boundaries_found = False
        self.boundaries = None
        self.boundary_background = None

        self.laser_center_calibrated = False
        self.laser_center = None
        self.laser_delta_x = 0.0
        self.laser_delta_y = 0.0

        self.LENGTH_HORIZONTAL = 20.0
        self.LENGTH_VERTICAL = 20.0

        # Add after: self.LENGTH_VERTICAL = 20.0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def get_line_distance_squared(self, line):
        x1, y1, x2, y2 = line
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    def merge_lines(self, lines, orientation='horizontal', tolerance=5):
        if not lines:
            return []

        if orientation == 'horizontal':
            lines = sorted(lines, key=lambda ln: (ln[1] + ln[3]) / 2)
        else:
            lines = sorted(lines, key=lambda ln: (ln[0] + ln[2]) / 2)

        merged = []
        for ln in lines:
            x1, y1, x2, y2 = ln
            mid = (y1 + y2) / 2 if orientation == 'horizontal' else (x1 + x2) / 2

            # after sorting by midpoint, slowly add lines to merge and re-merge them
            # this is very slow O(n^2) for lots of line but for our case is fine
            found = False
            for i, m in enumerate(merged):
                m_mid = (m[1] + m[3]) / 2 if orientation == 'horizontal' else (m[0] + m[2]) / 2
                if abs(mid - m_mid) <= tolerance:
                    # simple min max merge
                    merged[i] = (
                        min(x1, m[0]),
                        min(y1, m[1]),
                        max(x2, m[2]),
                        max(y2, m[3])
                    )
                    found = True
                    break
            if not found:
                merged.append(ln)

        return merged

    def reset_boundaries(self):
        self.boundaries = None
        self.boundaries_found = False

    def reset_center_calibration(self):
        self.laser_center = None
        self.laser_delta_x = 0.0
        self.laser_delta_y = 0.0
        self.laser_center_calibrated = False

    def find_boundaries(self, frame, p1, p2):

        if self.boundaries_found and self.boundaries:
            return self.boundaries

        x1, y1 = p1
        x2, y2 = p2

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # xmin, ymin = xmin + offset, ymin + offset
        # xmax, ymax = xmax - offset, ymax - offset

        # get the selected roi only
        roi = frame[ymin:ymax, xmin:xmax].copy()

        # convert to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        roi = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,  # size of neighborhood
            C=20  # offset from mean
        )

        cv2.imshow("boundaries_roi", roi)

        # count the 4 lines these many times correctly to be sure
        verification_count = 10
        # find at least 75% of consistent boundaries
        verification_min_consistent = int(3/4 * verification_count)

        current_iterations = 0
        max_iterations = verification_count * 2

        verification_samples = []

        while len(verification_samples) < verification_count and current_iterations <= max_iterations:
            current_iterations += 1

            horizontal = []
            vertical = []

            lines = cv2.HoughLinesP(
                roi,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=roi.shape[1] // 4,
                maxLineGap=2
            )

            # find at least 4 lines
            if lines is None or len(lines) < 4:
                continue

            # sort the lines in descending order
            # lines = sorted(lines, key=lambda ln: self.get_line_distance_squared(ln[0]), reverse=True)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1

                # roi is local, so convert to the original 'frame' coord system
                origin_line = (x1 + xmin, y1 + ymin, x2 + xmin, y2 + ymin)

                # for dy < dx, lines are mostly horizontal, else vertical
                if abs(dy) < abs(dx):
                    horizontal.append(origin_line)
                else:
                    vertical.append(origin_line)

            # merge duplicates first
            horizontal = self.merge_lines(horizontal, orientation='horizontal', tolerance=5)
            vertical = self.merge_lines(vertical, orientation='vertical', tolerance=5)

            # sort horizontals top to bottom, and verticals left to right
            horizontal = sorted(horizontal, key=lambda ln: (ln[1] + ln[3]) / 2)
            vertical = sorted(vertical, key=lambda ln: (ln[0] + ln[2]) / 2)

            # horizontals and verticals can have overlaps due to practical thickness
            # filter approximates


            # at least 2 horizontals and 2 verticals (4 total)
            if len(horizontal) < 2 or len(vertical) < 2:
                continue

            # after sorting, our lines should be the 2 biggest horizontal and 2 of the verticals
            left, right, top, bottom = vertical[0], vertical[1], horizontal[0], horizontal[1]

            verification_samples.append((left, right, top, bottom))



        # only keep samples where the length of all lines are consistent
        # conservative, but keep ones that dont exceed half the avg length
        consistent_samples = []
        for sample in verification_samples:
            lengths = [self.get_line_distance_squared(ln) for ln in sample]
            avg_length = np.mean(lengths)

            if all(abs(l - avg_length) <= avg_length / 2 for l in lengths):
                consistent_samples.append(sample)


        if len(consistent_samples) < verification_min_consistent:
            print("Couldn't ensure correct boundaries. Try a better ROI or increasing the number of samples taken.")
            return None

        avg_lines = np.mean(np.array(consistent_samples), axis=0, dtype=np.int32)

        left, right, top, bottom = [tuple(line) for line in avg_lines]

        # calculate the center of the boundaries
        x_center = ((left[0] + left[2]) / 2 + (right[0] + right[2]) / 2) / 2
        y_center = ((top[1] + top[3]) / 2 + (bottom[1] + bottom[3]) / 2) / 2

        boundaries = {'left': left, 'right': right, 'top': top, 'bottom': bottom}

        # boundaries found, update state
        self.boundaries_found = True
        self.boundaries = boundaries
        self.boundaries_center = (int(x_center), int(y_center))

        print("Boundaries found and cached.")
        print(self.boundaries)

        return boundaries

    def calibrate_center(self, frame, tolerance=10):
        if not self.boundaries or not self.boundaries_found:
            return None, None

        if self.laser_center_calibrated:
            return self.laser_center, self.boundaries_center


        left, right, top, bottom = itemgetter('left', 'right', 'top', 'bottom')(self.boundaries)

        x_center, y_center = self.boundaries_center

        roi = frame[y_center-tolerance : y_center+tolerance, x_center-tolerance : x_center+tolerance].copy()
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, roi = cv2.threshold(roi, 220, 255, cv2.THRESH_BINARY)
        cv2.imshow("laser roi", roi)

        active_ys, active_xs = np.where(roi == 255)

        if len(active_xs) == 0 or len(active_ys) == 0:
            print("No laser dot detected in given ROI.")
            return (x_center, y_center), (x_center, y_center)

        # find the local active laser center
        local_y, local_x = np.mean(active_ys), np.mean(active_xs)

        # calculate pixel diffs
        delta_y, delta_x = local_y - tolerance, local_x - tolerance

        # offset the original center with the deltas
        laser_y, laser_x = int(y_center + delta_y), int(x_center + delta_x)

        self.laser_center = (laser_x, laser_y)
        self.laser_delta_x = delta_x
        self.laser_delta_y = delta_y
        self.laser_center_calibrated = True

        return (laser_x, laser_y), (x_center, y_center)


        # steps:
        # slice only the boundaries roi
        # dynamically threshold the image on the white scale
        # get min and max boundaries of the active pixels mask
        # add the delta x and delta y of the laser center calibration
        # compute the analogous distances in cm given the init constants
        # return the laser boundary coordinates, the real analogous distances of the box in cm and the avg intensity

    def track_laser_boundaries(self, frame):

        if not self.boundaries_found or not self.laser_center_calibrated:
            print("Boundaries or Laser Center Calibration are incomplete.")
            return {}

        left, right, top, bottom = itemgetter('left', 'right', 'top', 'bottom')(self.boundaries)

        # bottom left -> top right
        xmin = min(left[0], left[2])
        xmax = max(right[0], right[2])
        ymin = min(top[1], top[3])
        ymax = max(bottom[1], bottom[3])

        laser_roi = frame[ymin:ymax, xmin:xmax].copy()
        roi_hsv = cv2.cvtColor(laser_roi, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(laser_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # red has dual ranges
        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 30, 30])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        cv2.imshow('red mask', red_mask)

        kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_red)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_red, iterations=2)


        if self.boundary_background is None:
            self.boundary_background = roi_gray

        _, roi_gray = cv2.threshold(roi_gray, 210, 255, cv2.THRESH_BINARY)

        roi_gray = cv2.medianBlur(roi_gray, 5)

        ys, xs = np.where(roi_gray == 255)

        if len(xs) == 0 or len(ys) == 0:
            print("No active laser pixels detected.")
            return {}

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # x_min, x_max = x_min + self.laser_delta_x, x_max + self.laser_delta_x
        # y_min, y_max = y_min + self.laser_delta_y, y_max + self.laser_delta_y

        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        bound_dist_x_px = abs(xmax - xmin)
        bound_dist_y_px = abs(ymax - ymin)
        laser_dist_x_px = abs(x_max - x_min)
        laser_dist_y_px = abs(y_max - y_min)

        laser_dist_x = laser_dist_x_px / bound_dist_x_px * self.LENGTH_HORIZONTAL
        laser_dist_y = laser_dist_y_px / bound_dist_y_px * self.LENGTH_VERTICAL

        print("Distance X:", laser_dist_x, "Distance Y:", laser_dist_y)

        roi_gray = cv2.rectangle(roi_gray, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imshow("roi_gray", roi_gray)

        offset_x_px = self.laser_delta_x
        offset_y_px = self.laser_delta_y

        offset_x_cm = offset_x_px / bound_dist_x_px * self.LENGTH_HORIZONTAL
        offset_y_cm = offset_y_px / bound_dist_y_px * self.LENGTH_VERTICAL

        laser_p1 = (xmin + x_min, ymin + y_min)
        laser_p2 = (xmin + x_max, ymin + y_max)

        return {
            'laser_dist_x': laser_dist_x,
            'laser_dist_y': laser_dist_y,
            'laser_p1': laser_p1,
            'laser_p2': laser_p2,
            'laser_offset_x': offset_x_cm,
            'laser_offset_y': offset_y_cm,
        }


    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass




def mouse_callback(event, x, y, flags, param):

    # holding down button
    if event == cv2.EVENT_LBUTTONDOWN:
        param['box_ready'] = False
        param['p1'] = (x, y)

        # when initially holding down, set both equal to avoid weird drawings
        param['p2'] = param['p1']

        param['selecting_box'] = True


    # moving mouse and holding down button
    elif event == cv2.EVENT_MOUSEMOVE and param['selecting_box']:

        # set the p2 (final point)
        param['p2'] = (x, y)


        # to avoid weird drawings, enable the selection after moving
        param['selecting_box'] = True


    # lifting the mouse means final box selection
    elif event == cv2.EVENT_LBUTTONUP:
        param['selecting_box'] = False
        param['box_ready'] = True

        # order from left to right
        if param['p2'][0] < param['p1'][0]:
            p1 = param['p1']
            param['p1'] = param['p2']
            param['p2'] = p1


        print("Box coords:", param['p1'], param['p2'])

async def sweep_audio(state, min_freq, max_freq, step, duration):
    """Run a continuous frequency sweep asynchronously until cancelled."""
    try:
        current = min_freq
        while True:
            # generate sine wave for this frequency
            t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
            wave = np.sin(2 * np.pi * current * t).astype(np.float32)

            sd.play(wave, samplerate=44100, blocking=False)

            await asyncio.sleep(duration)

            current += step
            if current > max_freq:
                current = min_freq

            # if paused externally
            if not state["is_playing"]:
                sd.stop()
                return

    except asyncio.CancelledError:
        sd.stop()
        return


async def main():
    cap = cv2.VideoCapture(0)

    MIN_FREQ = 100.0
    MAX_FREQ = 1000.0
    STEP = 1
    DURATION = 1.0

    current_freq = MIN_FREQ

    state = {'selecting_box': False,
             'box_ready': False,
             'p1': (-1, -1),
             'p2': (-1, -1),
             'selection_color': (0, 255, 255),
             'ready_color': (0, 255, 0),
             'is_playing': False,
             'sweep_task': None
             }

    cv2.namedWindow("Raw")
    cv2.setMouseCallback("Raw", mouse_callback, state)

    tracker = DynamicLaserTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()

        # FPS calculation
        tracker.frame_count += 1
        if time.time() - tracker.last_fps_time >= 1.0:
            tracker.fps = tracker.frame_count
            tracker.frame_count = 0
            tracker.last_fps_time = time.time()

        # HUD overlay background (smaller)
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)

        # Status indicators (more compact)
        status_y = 18
        cv2.putText(output_frame, f"FPS: {tracker.fps}", (8, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(output_frame, f"BOUNDS: {'OK' if tracker.boundaries_found else 'NO'}",
                    (8, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0) if tracker.boundaries_found else (0, 0, 255), 1)
        cv2.putText(output_frame, f"CALIB: {'OK' if tracker.laser_center_calibrated else 'NO'}",
                    (8, status_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0) if tracker.laser_center_calibrated else (0, 0, 255), 1)
        cv2.putText(output_frame, f"AUDIO: {'SWEEP' if state['is_playing'] else 'IDLE'}",
                    (8, status_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 0) if state['is_playing'] else (100, 100, 100), 1)

        if state['selecting_box'] or state['box_ready']:
            color = state['selection_color'] if state['selecting_box'] else state['ready_color']
            cv2.rectangle(output_frame, state['p1'], state['p2'], color, 2)

            # ROI dimensions
            roi_w = abs(state['p2'][0] - state['p1'][0])
            roi_h = abs(state['p2'][1] - state['p1'][1])
            cv2.putText(output_frame, f"ROI: {roi_w}x{roi_h}px",
                        (state['p1'][0], state['p1'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if state['box_ready']:
            lines = tracker.find_boundaries(frame, state['p1'], state['p2'])
            if lines:
                # Draw boundaries with labels
                for key, line in lines.items():
                    cv2.line(output_frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
                    mid_x = (line[0] + line[2]) // 2
                    mid_y = (line[1] + line[3]) // 2
                    cv2.putText(output_frame, key.upper(), (mid_x - 20, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # calibrate and compute laser center
                laser_center, boundary_center = tracker.calibrate_center(frame, tolerance=40)

                # Draw crosshair at centers
                cross_size = 10
                cv2.line(output_frame, (boundary_center[0] - cross_size, boundary_center[1]),
                         (boundary_center[0] + cross_size, boundary_center[1]), (255, 255, 0), 1)
                cv2.line(output_frame, (boundary_center[0], boundary_center[1] - cross_size),
                         (boundary_center[0], boundary_center[1] + cross_size), (255, 255, 0), 1)

                cv2.circle(output_frame, laser_center, 5, (0, 0, 150), 2)
                cv2.circle(output_frame, laser_center, 2, (255, 255, 255), -1)

            info = tracker.track_laser_boundaries(frame)
            if info:
                p1 = info['laser_p1']
                p2 = info['laser_p2']

                # draw tracking box with corner markers
                cv2.rectangle(output_frame, p1, p2, (255, 0, 0), 2)
                corner_len = 8
                for corner in [p1, (p2[0], p1[1]), p2, (p1[0], p2[1])]:
                    cv2.circle(output_frame, corner, 3, (0, 255, 255), -1)

                # Distance line with glow effect
                cv2.line(output_frame, boundary_center, laser_center, (0, 100, 0), 3, cv2.LINE_AA)
                cv2.line(output_frame, boundary_center, laser_center, (0, 255, 0), 1, cv2.LINE_AA)

                # distance in cm from offsets
                dx_cm = info['laser_offset_x']
                dy_cm = info['laser_offset_y']

                # Euclidean distance in cm
                dist_cm = (dx_cm ** 2 + dy_cm ** 2) ** 0.5

                # Enhanced distance display (smaller)
                mx = (boundary_center[0] + laser_center[0]) // 2
                my = (boundary_center[1] + laser_center[1]) // 2

                text = f"{dist_cm:.1f}cm"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

                # Background for text (smaller)
                bg_pt1 = (mx + 5, my - text_size[1] - 8)
                bg_pt2 = (mx + text_size[0] + 10, my - 3)
                cv2.rectangle(output_frame, bg_pt1, bg_pt2, (0, 0, 0), -1)
                cv2.rectangle(output_frame, bg_pt1, bg_pt2, (0, 255, 0), 1)

                cv2.putText(output_frame, text, (mx + 7, my - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


                # Enhanced measurement displays
                text_x = f"{info['laser_dist_x']:.1f}cm"
                text_y = f"{info['laser_dist_y']:.1f}cm"

                cx = (p1[0] + p2[0]) // 2
                cy = (p1[1] + p2[1]) // 2

                # X measurement with background (smaller)
                text_x_pos = (cx - 25, p1[1] - 8)
                text_size_x = cv2.getTextSize(text_x, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                cv2.rectangle(output_frame,
                              (text_x_pos[0] - 3, text_x_pos[1] - text_size_x[1] - 3),
                              (text_x_pos[0] + text_size_x[0] + 3, text_x_pos[1] + 3),
                              (0, 0, 0), -1)
                cv2.putText(output_frame, text_x, text_x_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)

                # Y measurement with background (smaller)
                text_y_pos = (p1[0] - 45, cy + 3)
                text_size_y = cv2.getTextSize(text_y, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                cv2.rectangle(output_frame,
                              (text_y_pos[0] - 3, text_y_pos[1] - text_size_y[1] - 3),
                              (text_y_pos[0] + text_size_y[0] + 3, text_y_pos[1] + 3),
                              (0, 0, 0), -1)
                cv2.putText(output_frame, text_y, text_y_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)

                # Additional telemetry overlay
                telemetry_y = output_frame.shape[0] - 80
                cv2.rectangle(output_frame, (0, telemetry_y - 10),
                              (250, output_frame.shape[0]), (0, 0, 0), -1)
                cv2.rectangle(output_frame, (0, telemetry_y - 10),
                              (250, output_frame.shape[0]), (0, 100, 100), 2)

                cv2.putText(output_frame, "TELEMETRY", (10, telemetry_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(output_frame, f"Delta X: {dx_cm:+.2f} cm",
                            (10, telemetry_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.putText(output_frame, f"Delta Y: {dy_cm:+.2f} cm",
                            (10, telemetry_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.putText(output_frame, f"Area: {info['laser_dist_x'] * info['laser_dist_y']:.2f} cm^2",
                            (10, telemetry_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        else:
            tracker.reset_boundaries()
            tracker.reset_center_calibration()

            # Instruction overlay when no selection
            inst_text = "Draw ROI with mouse | SPACE: toggle audio | ESC: quit"
            cv2.putText(output_frame, inst_text, (10, output_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Raw", output_frame)

        key = cv2.waitKey(1) & 0xFF
        await asyncio.sleep(0.001)

        if key == 27:  # Esc
            break
        elif key == ord(' '):
            if not state['is_playing']:
                print("Starting sweep...")

                state['is_playing'] = True

                # create async sweep task
                state['sweep_task'] = asyncio.create_task(
                    sweep_audio(state, MIN_FREQ, MAX_FREQ, STEP, DURATION)
                )
            else:
                print("Stopping sweep...")
                state['is_playing'] = False

                # cancel sweep task
                if state['sweep_task']:
                    state['sweep_task'].cancel()
                    state['sweep_task'] = None
                sd.stop()


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
