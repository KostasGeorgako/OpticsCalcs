"""
Dynamic Laser Tracker + Async Audio Sweep
-----------------------------------------
Cleaner, more modular, more expressive version of your original script.
Behaves exactly the same, but far easier to maintain and extend.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import asyncio
import sounddevice as sd
from scipy import signal
from operator import itemgetter


# =============================================================================
# Utility helpers
# =============================================================================

def midpoint(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def draw_text(
    frame,
    text: str,
    origin: Tuple[int, int],
    color=(0, 255, 0),
    scale=0.5,
    thickness=1,
):
    cv2.putText(
        frame, text, origin,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


# =============================================================================
# CLASS: Dynamic Laser Tracker
# =============================================================================

class DynamicLaserTracker:
    """
    Handles:
    - Boundary detection via Hough lines
    - Laser center calibration
    - Laser boundary tracking + scale conversion
    """

    LENGTH_HORIZONTAL = 20.0
    LENGTH_VERTICAL = 20.0

    def __init__(self):
        # system state
        self.boundaries_found: bool = False
        self.boundaries: Optional[Dict[str, Tuple[int, int, int, int]]] = None

        self.laser_center_calibrated: bool = False
        self.laser_center: Optional[Tuple[int, int]] = None

        # deltas in px measured during center calibration
        self.laser_delta_x = 0.0
        self.laser_delta_y = 0.0

        self.boundary_background = None

    # -------------------------------------------------------------------------

    @staticmethod
    def get_line_distance_squared(line):
        x1, y1, x2, y2 = line
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    # -------------------------------------------------------------------------

    def merge_lines(self, lines, orientation='horizontal', tolerance=5):
        """Merge similar Hough lines (slow O(n²) is fine for small n)."""
        if not lines:
            return []

        # sort by midpoint
        if orientation == 'horizontal':
            lines = sorted(lines, key=lambda ln: (ln[1] + ln[3]) / 2)
        else:
            lines = sorted(lines, key=lambda ln: (ln[0] + ln[2]) / 2)

        merged = []
        for ln in lines:
            x1, y1, x2, y2 = ln
            mid = (y1 + y2) / 2 if orientation == 'horizontal' else (x1 + x2) / 2

            for i, m in enumerate(merged):
                m_mid = (m[1] + m[3]) / 2 if orientation == 'horizontal' else (m[0] + m[2]) / 2

                if abs(mid - m_mid) <= tolerance:
                    merged[i] = (
                        min(x1, m[0]),
                        min(y1, m[1]),
                        max(x2, m[2]),
                        max(y2, m[3])
                    )
                    break
            else:
                merged.append(ln)

        return merged

    # -------------------------------------------------------------------------

    def reset_boundaries(self):
        self.boundaries = None
        self.boundaries_found = False

    def reset_center_calibration(self):
        self.laser_center = None
        self.laser_center_calibrated = False
        self.laser_delta_x = 0.0
        self.laser_delta_y = 0.0

    # -------------------------------------------------------------------------
    # BOUNDARY DETECTION
    # -------------------------------------------------------------------------

    def find_boundaries(self, frame, p1, p2):
        """Detect top/bottom/left/right drawn boundary lines using Hough."""
        if self.boundaries_found:
            return self.boundaries

        x1, y1 = p1
        x2, y2 = p2
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))

        # ROI extraction
        roi_gray = cv2.cvtColor(frame[ymin:ymax, xmin:xmax].copy(), cv2.COLOR_BGR2GRAY)

        roi_bin = cv2.adaptiveThreshold(
            roi_gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=20
        )
        cv2.imshow("boundaries_roi", roi_bin)

        required = 10
        min_consistent = 8
        samples = []
        attempts = 0

        while len(samples) < required and attempts < required * 2:
            attempts += 1

            lines = cv2.HoughLinesP(
                roi_bin, rho=1, theta=np.pi/180,
                threshold=30,
                minLineLength=roi_bin.shape[1] // 4,
                maxLineGap=2
            )
            if lines is None or len(lines) < 4:
                continue

            horizontal, vertical = [], []

            for (x1r, y1r, x2r, y2r) in lines[:, 0]:
                dx = x2r - x1r
                dy = y2r - y1r

                # convert to global coords
                line_global = (x1r + xmin, y1r + ymin, x2r + xmin, y2r + ymin)

                (horizontal if abs(dy) < abs(dx) else vertical).append(line_global)

            horizontal = self.merge_lines(horizontal, 'horizontal')
            vertical = self.merge_lines(vertical, 'vertical')

            if len(horizontal) < 2 or len(vertical) < 2:
                continue

            horizontal.sort(key=lambda ln: (ln[1] + ln[3]) / 2)
            vertical.sort(key=lambda ln: (ln[0] + ln[2]) / 2)

            samples.append((vertical[0], vertical[1], horizontal[0], horizontal[1]))

        # Validate consistency
        consistent = []
        for s in samples:
            lengths = [self.get_line_distance_squared(ln) for ln in s]
            avg_len = np.mean(lengths)
            if all(abs(l - avg_len) <= avg_len * 0.5 for l in lengths):
                consistent.append(s)

        if len(consistent) < min_consistent:
            print("Boundary detection failed.")
            return None

        # Average
        avg = np.mean(np.array(consistent), axis=0, dtype=np.int32)
        left, right, top, bottom = [tuple(line) for line in avg]

        # Compute center
        x_center = ((left[0] + left[2]) / 2 + (right[0] + right[2]) / 2) / 2
        y_center = ((top[1] + top[3]) / 2 + (bottom[1] + bottom[3]) / 2) / 2

        self.boundaries_found = True
        self.boundaries = {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
        }
        self.boundaries_center = (int(x_center), int(y_center))

        print("Boundaries found.")
        return self.boundaries

    # -------------------------------------------------------------------------
    # LASER CENTER CALIBRATION
    # -------------------------------------------------------------------------

    def calibrate_center(self, frame, tolerance=10):
        if not self.boundaries_found:
            return None, None

        if self.laser_center_calibrated:
            return self.laser_center, self.boundaries_center

        bx, by = self.boundaries_center
        roi = frame[by - tolerance:by + tolerance, bx - tolerance:bx + tolerance]
        if roi.size == 0:
            return None, None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_bin = cv2.threshold(roi_gray, 220, 255, cv2.THRESH_BINARY)

        cv2.imshow("laser_roi", roi_bin)

        ys, xs = np.where(roi_bin == 255)
        if len(xs) == 0:
            return (bx, by), (bx, by)

        local_x = np.mean(xs)
        local_y = np.mean(ys)

        dx = local_x - tolerance
        dy = local_y - tolerance

        self.laser_center = (int(bx + dx), int(by + dy))
        self.laser_delta_x = dx
        self.laser_delta_y = dy
        self.laser_center_calibrated = True

        return self.laser_center, self.boundaries_center

    # -------------------------------------------------------------------------
    # LASER BOUNDARY TRACKING
    # -------------------------------------------------------------------------

    def track_laser_boundaries(self, frame) -> Dict:
        if not (self.boundaries_found and self.laser_center_calibrated):
            return {}

        left, right, top, bottom = itemgetter('left', 'right', 'top', 'bottom')(self.boundaries)

        xmin = min(left[0], left[2])
        xmax = max(right[0], right[2])
        ymin = min(top[1], top[3])
        ymax = max(bottom[1], bottom[3])

        roi = frame[ymin:ymax, xmin:xmax]
        roi_gray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)

        if self.boundary_background is None:
            self.boundary_background = roi_gray

        _, roi_bin = cv2.threshold(roi_gray, 210, 255, cv2.THRESH_BINARY)
        roi_bin = cv2.medianBlur(roi_bin, 5)

        ys, xs = np.where(roi_bin == 255)
        if len(xs) == 0:
            return {}

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Distances in px within boundary box
        bound_dx = xmax - xmin
        bound_dy = ymax - ymin

        laser_dx = x_max - x_min
        laser_dy = y_max - y_min

        # Convert to cm
        dist_x = laser_dx / bound_dx * self.LENGTH_HORIZONTAL
        dist_y = laser_dy / bound_dy * self.LENGTH_VERTICAL

        # Offset in cm
        off_x = self.laser_delta_x / bound_dx * self.LENGTH_HORIZONTAL
        off_y = self.laser_delta_y / bound_dy * self.LENGTH_VERTICAL

        return {
            'laser_dist_x': dist_x,
            'laser_dist_y': dist_y,
            'laser_p1': (xmin + x_min, ymin + y_min),
            'laser_p2': (xmin + x_max, ymin + y_max),
            'laser_offset_x': off_x,
            'laser_offset_y': off_y,
        }


# =============================================================================
# MOUSE CALLBACK
# =============================================================================

def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state['box_ready'] = False
        state['selecting_box'] = True
        state['p1'] = (x, y)
        state['p2'] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and state['selecting_box']:
        state['p2'] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state['selecting_box'] = False
        state['box_ready'] = True

        if state['p2'][0] < state['p1'][0]:
            state['p1'], state['p2'] = state['p2'], state['p1']

        print("ROI selected:", state['p1'], state['p2'])


# =============================================================================
# AUDIO SWEEP (ASYNC)
# =============================================================================

async def sweep_audio(state, min_freq, max_freq, step, duration):
    """Continuous frequency sweep until cancelled."""
    current = min_freq
    try:
        while True:
            t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
            wave = np.sin(2 * np.pi * current * t).astype(np.float32)

            sd.play(wave, samplerate=44100, blocking=False)
            await asyncio.sleep(duration)

            current += step
            if current > max_freq:
                current = min_freq

            if not state['is_playing']:
                break

    except asyncio.CancelledError:
        pass

    finally:
        sd.stop()


# =============================================================================
# MAIN LOOP
# =============================================================================

async def main():
    cap = cv2.VideoCapture(0)

    MIN_FREQ = 100
    MAX_FREQ = 1000
    STEP = 1
    DURATION = 1.0

    state = {
        'selecting_box': False,
        'box_ready': False,
        'p1': (-1, -1),
        'p2': (-1, -1),
        'selection_color': (0, 255, 255),
        'ready_color': (0, 255, 0),
        'is_playing': False,
        'sweep_task': None,
    }

    cv2.namedWindow("Raw")
    cv2.setMouseCallback("Raw", mouse_callback, state)

    tracker = DynamicLaserTracker()

    # ---------------------------------------------------------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw = frame.copy()

        # Draw box selection
        if state['selecting_box'] or state['box_ready']:
            color = state['selection_color'] if state['selecting_box'] else state['ready_color']
            cv2.rectangle(draw, state['p1'], state['p2'], color, 2)

        # -----------------------------------------------------------------
        if state['box_ready']:
            boundaries = tracker.find_boundaries(frame, state['p1'], state['p2'])

            if boundaries:
                for ln in boundaries.values():
                    cv2.line(draw, (ln[0], ln[1]), (ln[2], ln[3]), (0, 0, 255), 1)

            laser_center, boundary_center = tracker.calibrate_center(frame, 40)
            if laser_center:
                cv2.circle(draw, laser_center, 3, (0, 0, 200), -1)

            info = tracker.track_laser_boundaries(frame)
            if info:
                p1, p2 = info['laser_p1'], info['laser_p2']
                cv2.rectangle(draw, p1, p2, (255, 0, 0), 2)

                # connection line
                cv2.line(draw, boundary_center, laser_center, (0, 255, 0), 1, cv2.LINE_AA)

                # cm distance
                dx = info['laser_offset_x']
                dy = info['laser_offset_y']
                dist = (dx**2 + dy**2)**0.5

                mid = midpoint(boundary_center, laser_center)
                draw_text(draw, f"{dist:.1f} cm", (mid[0] + 15, mid[1] - 15))

                # box dimensions
                bx = f"{info['laser_dist_x']:.1f} cm"
                by = f"{info['laser_dist_y']:.1f} cm"

                draw_text(draw, bx, (p1[0], p1[1] - 10), (255, 0, 0))
                draw_text(draw, by, (p1[0] - 70, (p1[1] + p2[1]) // 2), (255, 0, 0))

        else:
            tracker.reset_boundaries()
            tracker.reset_center_calibration()

        # -----------------------------------------------------------------

        cv2.imshow("Raw", draw)

        key = cv2.waitKey(1) & 0xFF
        await asyncio.sleep(0)

        if key == 27:  # ESC
            break

        elif key == ord(' '):
            if not state['is_playing']:
                print("Starting sweep...")
                state['is_playing'] = True
                state['sweep_task'] = asyncio.create_task(
                    sweep_audio(state, MIN_FREQ, MAX_FREQ, STEP, DURATION)
                )
            else:
                print("Stopping sweep...")
                state['is_playing'] = False
                if state['sweep_task']:
                    state['sweep_task'].cancel()
                    state['sweep_task'] = None

    # ---------------------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================

if __name__ == "__main__":
    asyncio.run(main())
