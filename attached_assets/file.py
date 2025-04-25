# -*- coding: utf-8 -*-
# Required libraries
import cv2
import mediapipe as mp
import time
import math # Needed for math.ceil
import numpy as np
import traceback
import webbrowser
import urllib.parse
import pygame
# No deque needed

# --- Configuration ---
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Keyboard Layout Parameters ---
KEYBOARD_START_X = 10; KEYBOARD_START_Y = 10
KEY_WIDTH = 50; KEY_HEIGHT = 50; KEY_GAP = 5
NORMAL_KEYS_PER_ROW = 5
SPECIAL_KEYS = ["<-", "SPC", "RST", "SRCH"]
KEYS_PER_ROW_TOTAL = NORMAL_KEYS_PER_ROW + len(SPECIAL_KEYS)

# --- Timing ---
TRAVERSAL_INTERVAL = 0.6
POST_BLINK_PAUSE = 0.5
INITIAL_DELAY = 10.0

# --- Blink Detection Settings ---
EAR_THRESHOLD = 0.20  # !! TUNE THIS !!
CONSECUTIVE_FRAMES_THRESHOLD = 2

# --- Colors ---
KEY_COLOR = (215, 215, 215); KEY_TEXT_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (180, 255, 180); ROW_HIGHLIGHT_COLOR = (200, 220, 255)
SPECIAL_KEY_COLOR = (200, 200, 230); TYPED_TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (100, 100, 100)
ICON_COLOR = (0, 0, 0)

# --- Scanning Modes ---
SCANNING_MODE_ROW = 0; SCANNING_MODE_KEY = 1

# --- Global Variables & Layout ---
# (Layout generation code - same as previous version, no dummy key)
keyboard_rows = []
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letter_idx = 0; total_keys_in_layout = 0
print("Generating Keyboard Layout (No Dummy Key)...")
num_letter_rows = math.ceil(len(letters) / NORMAL_KEYS_PER_ROW)
for current_row_num in range(num_letter_rows):
    row_keys_chars = []; current_row_layout = []
    for _ in range(NORMAL_KEYS_PER_ROW):
        if letter_idx < len(letters): row_keys_chars.append(letters[letter_idx]); letter_idx += 1
        else: row_keys_chars.append(None)
    row_keys_chars.extend(SPECIAL_KEYS)
    for col_idx, char in enumerate(row_keys_chars):
        if char is not None:
            x=KEYBOARD_START_X+col_idx*(KEY_WIDTH+KEY_GAP); y=KEYBOARD_START_Y+current_row_num*(KEY_HEIGHT+KEY_GAP)
            keyboard_data = {"char": char, "x": x, "y": y, "w": KEY_WIDTH, "h": KEY_HEIGHT, "row": current_row_num, "col": col_idx}
            current_row_layout.append(keyboard_data); total_keys_in_layout += 1
    if current_row_layout: keyboard_rows.append(current_row_layout)
num_rows_actual = len(keyboard_rows)
print(f"Layout generated: {num_rows_actual} rows, {total_keys_in_layout} total keys.")
print("-" * 20)

# --- State & Timing Variables ---
current_scan_mode = SCANNING_MODE_ROW; current_row_scan_index = 0
current_key_scan_index_in_row = 0; selected_row_index = -1
last_traversal_time = time.time(); typed_text = ""
last_blink_time = 0
consecutive_blink_frames = 0
is_paused = False
frame_size_printed = False
initial_delay_passed = False; start_time = time.time()

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh; mp_drawing = mp.solutions.drawing_utils
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]; RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Auditory Feedback Initialization ---
pygame.mixer.init()
action_sound = None
selection_sound = None # Initialize new sound variable

# --- Load Key Action Sound ---
try:
    action_sound_path = "C:\\Users\\Afiya Khadir\\Downloads\\action.wav" # !!! REPLACE FOR KEY SELECTION SOUND !!!
    print(f"Loading ACTION sound from: {action_sound_path}")
    action_sound = pygame.mixer.Sound(action_sound_path)
except Exception as e: print(f"Error loading ACTION sound: {e}. No action sound feedback."); action_sound = None

# --- Load Row Selection Sound ---
try:
    # !!! IMPORTANT: SET PATH TO YOUR ROW SELECTION SOUND FILE !!!
    selection_sound_path = "C:\\Users\\Afiya Khadir\\Downloads\\mixkit-correct-answer-tone-2870.wav" # !!! REPLACE FOR ROW SELECTION SOUND !!!
    print(f"Loading SELECTION sound from: {selection_sound_path}")
    selection_sound = pygame.mixer.Sound(selection_sound_path)
except Exception as e: print(f"Error loading SELECTION sound: {e}. No selection sound feedback."); selection_sound = None


# --- Helper Functions ---
# (play_sound, calculate_distance, calculate_ear - same as before)
def play_sound(sound):
    if sound:
        try: sound.play()
        except pygame.error as e: print(f"Error playing sound: {e}")

def calculate_distance(p1, p2):
    try:
        if p1 is None or p2 is None or len(p1)<2 or len(p2)<2: return 0.0
        if not (math.isfinite(p1[0]) and math.isfinite(p1[1]) and math.isfinite(p2[0]) and math.isfinite(p2[1])): return 0.0
        dx=p1[0]-p2[0]; dy=p1[1]-p2[1]; sq_dist=dx**2+dy**2
        return math.sqrt(sq_dist) if sq_dist>=0 else 0.0
    except Exception: return 0.0

def calculate_ear(eye_landmarks, frame_shape):
    try:
        if not eye_landmarks or len(eye_landmarks)!=6: return 0.6
        coords_list=[(lm.x*frame_shape[1], lm.y*frame_shape[0]) if (lm and hasattr(lm,'x') and hasattr(lm,'y') and math.isfinite(lm.x) and math.isfinite(lm.y)) else (np.nan,np.nan) for lm in eye_landmarks]
        coords=np.array(coords_list);
        if np.isnan(coords).any(): return 0.6
        v1=calculate_distance(coords[1],coords[5]); v2=calculate_distance(coords[2],coords[4]); h1=calculate_distance(coords[0],coords[3])
        if h1<=1e-6: return 0.6
        ear=(v1+v2)/(2.0*h1); return ear if math.isfinite(ear) else 0.6
    except Exception: return 0.6

# --- Main Program ---
cap = None
try:
    # --- Initialization ---
    print(f"Script started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    actual_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH); actual_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested: {FRAME_WIDTH}x{FRAME_HEIGHT}, Actual: {int(actual_width)}x{int(actual_height)}")

    print("Initializing MediaPipe Face Mesh...")
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        print(f"Initialization complete. Waiting for {INITIAL_DELAY:.1f}s delay...")
        print("Blink eye to select ROW or KEY.")
        print("Press 'q' in the OpenCV window to quit.")
        print("-" * 20)

        while cap.isOpened():
            current_time = time.time()

            try:
                success, frame = cap.read()
                if not success or frame is None: time.sleep(0.1); continue
                if not frame_size_printed: print(f"Proc Frame: {frame.shape[1]}x{frame.shape[0]}"); frame_size_printed=True

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)

                ear_avg = 0.6; face_detected = False; blink_detected_this_frame = False

                if results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    # --- EAR Calculation & Blink Detection ---
                    left_ear=calculate_ear([face_landmarks[i] for i in LEFT_EYE_INDICES],frame.shape)
                    right_ear=calculate_ear([face_landmarks[i] for i in RIGHT_EYE_INDICES], frame.shape)
                    ear_avg=(left_ear+right_ear)/2.0
                    if ear_avg<EAR_THRESHOLD: consecutive_blink_frames+=1
                    else:
                        if consecutive_blink_frames>=CONSECUTIVE_FRAMES_THRESHOLD:
                            if not is_paused and initial_delay_passed: blink_detected_this_frame=True
                        consecutive_blink_frames=0
                else: consecutive_blink_frames = 0

                # --- Pause Handling ---
                if is_paused and current_time - last_blink_time >= POST_BLINK_PAUSE:
                    is_paused = False; last_traversal_time = current_time

                # --- Initial Delay ---
                if not initial_delay_passed:
                    if current_time - start_time >= INITIAL_DELAY:
                        initial_delay_passed=True; last_traversal_time=current_time
                        print("-" * 20); print("--- Delay passed. Scanning starts. ---"); print("-" * 20)
                        current_scan_mode=SCANNING_MODE_ROW; current_row_scan_index=0; selected_row_index=-1; current_key_scan_index_in_row=0

                # --- Blink Action Logic ---
                if blink_detected_this_frame:
                    # --- MODIFIED: Moved sound playing inside modes ---
                    # play_sound(action_sound) # Removed from here
                    last_blink_time = current_time
                    is_paused = True

                    if current_scan_mode == SCANNING_MODE_ROW:
                        # --- Row Selected by Blink ---
                        if 0 <= current_row_scan_index < num_rows_actual:
                            selected_row_index = current_row_scan_index
                            print(f"Blink: Row {selected_row_index} selected.")
                            current_scan_mode = SCANNING_MODE_KEY
                            current_key_scan_index_in_row = 0
                            last_traversal_time = current_time
                            # --- MODIFIED: Play SELECTION sound here ---
                            play_sound(selection_sound)
                            print("  -> Key Scan mode.")
                        else:
                            print(f"Warn: Invalid row {current_row_scan_index}"); is_paused = False # Don't play sound if invalid

                    elif current_scan_mode == SCANNING_MODE_KEY:
                        # --- Key Selected by Blink ---
                        action_taken = False
                        if 0 <= selected_row_index < num_rows_actual:
                            keys_in_row = keyboard_rows[selected_row_index]
                            if 0 <= current_key_scan_index_in_row < len(keys_in_row):
                                key_info = keys_in_row[current_key_scan_index_in_row]; char = key_info["char"]
                                print(f"Blink: Key '{char}' (R:{selected_row_index},C:{key_info['col']}) selected.", end="")
                                # --- MODIFIED: Play ACTION sound here ---
                                play_sound(action_sound)
                                # --- Perform Key Action ---
                                if char == "<-": typed_text=typed_text[:-1]; print(" -> Bksp"); action_taken=True
                                elif char == "SPC": typed_text+=" "; print(" -> Spc"); action_taken=True
                                elif char == "RST": current_key_scan_index_in_row=0; last_traversal_time=current_time; print(" -> LineRst"); action_taken=False
                                elif char == "SRCH":
                                    if typed_text:
                                        try: query=urllib.parse.quote_plus(typed_text); url=f"https://www.google.com/search?q={query}"; print(f" -> Search '{typed_text}'..."); webbrowser.open(url,new=2)
                                        except Exception as e: print(f" Err: {e}")
                                    else: print(" -> SrchSkip")
                                    action_taken=True
                                else: typed_text+=char; print(f" -> Type '{char}'"); action_taken=True
                                # --- Reset To Row Scan? ---
                                if action_taken: current_scan_mode=SCANNING_MODE_ROW; current_row_scan_index=0; selected_row_index=-1; current_key_scan_index_in_row=0; last_traversal_time=current_time; print(" -> To RowScan")
                            else:
                                print(f"\nErr: Inv key idx {current_key_scan_index_in_row}");
                                is_paused=False # Don't play sound if invalid key
                        else:
                            print(f"\nErr: Inv row idx {selected_row_index}");
                            is_paused=False # Don't play sound if invalid row


                # --- Traversal Logic ---
                if initial_delay_passed and not is_paused:
                    if current_time - last_traversal_time >= TRAVERSAL_INTERVAL:
                        # (Same traversal logic)
                        if current_scan_mode==SCANNING_MODE_ROW: current_row_scan_index=(current_row_scan_index+1)%num_rows_actual
                        elif current_scan_mode==SCANNING_MODE_KEY:
                            if 0<=selected_row_index<num_rows_actual:
                                keys_in_row=keyboard_rows[selected_row_index]
                                if keys_in_row: current_key_scan_index_in_row=(current_key_scan_index_in_row+1)%len(keys_in_row)
                                else: current_scan_mode=SCANNING_MODE_ROW; current_row_scan_index=0; selected_row_index=-1
                            else: current_scan_mode=SCANNING_MODE_ROW; current_row_scan_index=0; selected_row_index=-1
                        last_traversal_time = current_time

                # --- Drawing ---
                # (Drawing logic remains the same as previous version)
                for r_idx, row in enumerate(keyboard_rows):
                    for k_idx, key in enumerate(row):
                        if key["x"]+key["w"]<frame.shape[1] and key["y"]+key["h"]<frame.shape[0]:
                            highlight=False; base_color=SPECIAL_KEY_COLOR if key["char"] in SPECIAL_KEYS else KEY_COLOR
                            color=base_color
                            if current_scan_mode == SCANNING_MODE_ROW:
                                if r_idx == current_row_scan_index: color=ROW_HIGHLIGHT_COLOR; highlight=True
                            elif current_scan_mode == SCANNING_MODE_KEY:
                                if r_idx == selected_row_index:
                                    if k_idx == current_key_scan_index_in_row: color=HIGHLIGHT_COLOR; highlight=True
                            cv2.rectangle(frame, (key["x"],key["y"]), (key["x"]+key["w"],key["y"]+key["h"]), color, -1)
                            cv2.rectangle(frame, (key["x"],key["y"]), (key["x"]+key["w"],key["y"]+key["h"]), BORDER_COLOR, 1)
                            char=key["char"]
                            if char == "SRCH":
                                cx=key["x"]+key["w"]//2; cy=key["y"]+key["h"]//2; radius=min(key["w"],key["h"])//4
                                handle_len=int(radius*1.2); thickness=2; circle_cx=cx-radius//3; circle_cy=cy-radius//3
                                cv2.circle(frame, (circle_cx, circle_cy), radius, ICON_COLOR, thickness)
                                angle_rad=math.radians(45); line_sx=circle_cx+int(radius*math.cos(angle_rad)); line_sy=circle_cy+int(radius*math.sin(angle_rad))
                                line_ex=line_sx+int(handle_len*math.cos(angle_rad)); line_ey=line_sy+int(handle_len*math.sin(angle_rad))
                                cv2.line(frame, (line_sx, line_sy), (line_ex, line_ey), ICON_COLOR, thickness + 1)
                            else:
                                font_scale=0.7
                                if len(char)==1: font_scale=0.7
                                elif len(char)<=3: font_scale=0.6
                                else: font_scale=0.6
                                if font_scale>0:
                                    thick=1; t_size,_=cv2.getTextSize(char,cv2.FONT_HERSHEY_SIMPLEX,font_scale,thick)
                                    t_x=key["x"]+(key["w"]-t_size[0])//2; t_y=key["y"]+(key["h"]+t_size[1])//2
                                    cv2.putText(frame, char, (t_x,t_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, KEY_TEXT_COLOR, thick)

                # Draw Typed Text Area
                typed_text_Y = KEYBOARD_START_Y+num_rows_actual*(KEY_HEIGHT+KEY_GAP)+30
                text_bg_y_start=typed_text_Y-25; text_bg_y_end=typed_text_Y+10
                if text_bg_y_end<frame.shape[0]:
                    cv2.rectangle(frame,(KEYBOARD_START_X-5,text_bg_y_start),(frame.shape[1]-10,text_bg_y_end),(50,50,50),-1)
                    cv2.putText(frame, "Typed: "+typed_text,(KEYBOARD_START_X,typed_text_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,TYPED_TEXT_COLOR,2)

                # --- REMOVED ALL STATUS TEXT OVERLAYS ---

                # --- Show Frame ---
                cv2.imshow('Blink Keyboard - Q to Quit', frame)

                # --- Exit Condition ---
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'): print("'q' pressed. Exiting loop..."); break

            except Exception as loop_error: print(f"\nERROR in loop: {loop_error}"); traceback.print_exc(); time.sleep(0.5)

except Exception as e: print(f"\nCRITICAL ERROR: {e}"); traceback.print_exc()
finally:
    # --- Cleanup ---
    print("-"*20); print("Executing cleanup...")
    if cap is not None and cap.isOpened(): print("Releasing camera..."); cap.release()
    print("Destroying OpenCV windows..."); cv2.destroyAllWindows(); cv2.waitKey(1)
    print("Quitting Pygame mixer..."); pygame.mixer.quit()
    print("Script finished."); print("-"*20)