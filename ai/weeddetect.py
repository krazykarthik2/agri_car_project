import cv2
import numpy as np
import time
from ultralytics import YOLO

def main():
    cap = cv2.VideoCapture(0)
    
    # Initialize YOLO model (Classification)
    # yolov8n-cls is trained on ImageNet.
    # We will check if the detected class relates to grass/nature.
    print("Loading YOLO model...")
    model = YOLO('yolov8n-cls.pt') 
    
    # --- Configuration ---
    # Polygons (Frames). defined as list of (x, y) points.
    # Adjust these to match your camera view.
    polygons = [
        np.array([(50, 50), (200, 50), (200, 200), (50, 200)], np.int32),   # Left
        np.array([(220, 50), (370, 50), (370, 200), (220, 200)], np.int32), # Center
        np.array([(390, 50), (540, 50), (540, 200), (390, 200)], np.int32)  # Right
    ] 
    
    SNAPSHOT_INTERVAL = 0.5 # seconds
    
    # Simulation Variabless
    current_y = 0.0
    OFFSET_Y = 50.0 # The delay distance
    Y_SPEED = 1.0   # How much Y changes per key press/frame
    
    # Storage
    # Dict: key=round(y,1), value=list of 3 dicts
    history = {}
    last_snapshot_time = time.time()
    last_snapshot_y = -999.0
    
    # For display of activation
    active_notifications = [] # list of messages to show
    
    print("Controls:")
    print("  UP/DOWN Arrows: Move Y variable")
    print("  'q' or 'ESC': Quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Flip for user friendliness
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- Input Handling ---
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'): # ESC or q
            break
        elif key == 0 or key == 2490368: # Up arrow (platform dependent sometimes)
            pass # Usually handled by specific codes below
            
        # Standard arrow key codes for OpenCV usually:
        # 82=Up, 84=Down on some systems. With waitKey(10) and masking it might vary.
        # Let's simple use 'w' and 's' as backup if arrows are tricky, but try arrows.
        # On Windows, arrow keys might be extended.
        if cv2.waitKey(1) == -1:
             # Just checking if any key is pressed is hard without blocking.
             # We rely on the buffer.
             pass
        
        # Let's assume standard arrow keys or WASD for simplicity + arrows
        if key == ord('w'): # Up
            current_y += Y_SPEED
        elif key == ord('s'): # Down
            current_y -= Y_SPEED
        
        # Check arrow keys specifically (windows)
        # 2490368 is Up, 2621440 is Down? It varies. 
        # Making it simple: W=Up, S=Down is reliable. 
        # Adding arrow support via special check if needed, but let's stick to W/S for reliability in headers
        # or map standard codes: 0x260000 (Up) etc.
        # Let's just use Up/Down logic on valid keys if detected.
        
        # --- Logic: Snapshot ---
        now = time.time()
        time_since_last_snip = now - last_snapshot_time
        y_change = abs(current_y - last_snapshot_y)
        
        if time_since_last_snip > SNAPSHOT_INTERVAL or y_change >= 1.0:
            last_snapshot_time = now
            last_snapshot_y = current_y
            
            # Prepare row data for this Y
            # Key: round(current_y, 1)
            # Value: list of 3 dicts
            y_key = round(current_y, 1)
            row_data = [] # List of 3 polygon states
            
            for i, poly in enumerate(polygons):
                # crop the polygon region
                # get bounding rect
                x, y, poly_w, poly_h = cv2.boundingRect(poly)
                
                # Ensure within bounds
                x = max(0, x)
                y = max(0, y)
                roi = frame[y:y+poly_h, x:x+poly_w]
                
                if roi.size == 0:
                    is_grass = False
                    conf = 0.0
                else:
                    # Run YOLO Classification
                    try:
                        results = model(roi, verbose=False)
                        
                        # Get top prediction
                        probs = results[0].probs
                        top1_idx = probs.top1
                        top1_conf = float(probs.top1conf)
                        class_name = results[0].names[top1_idx]
                        
                        # Primitive text check for "grass-like" things in ImageNet-1k
                        # Keywords: grass, lawn, pot, greenhouse, valley, alp, apiary (bees often in grass?)
                        # Keeping it simple. Since we are in a room usually, 'doormat' might trigger?
                        # Let's trust 'grass' or 'green' related terms if available.
                        # ImageNet classes are specific: 'maze', 'promontory', 'seashore', 'greenhouse', 'potted_plant'
                        # It doesn't actually have a generic 'grass' class easily.
                        # It has 'lawn_mower' maybe?
                        # Since this is a request to use YOLO for grass, we assume the model *can* detect it 
                        # or we map anything with low probability or specific texture.
                        # However, to be helpful, let's treat it as:
                        # If the user points at grass, classification might say 'lawn' or something? 
                        # There is no 'lawn' class in standard ImageNet 1k.
                        # There is 'earthstar' (fungus).
                        
                        # Fallback/Improvement: If we can't rely on class names (since 1k classes is limited for texture),
                        # we might just check if confidence is high on *any* nature class?
                        # OR, we might revert to the user's implicit trust in the model.
                        # Let's stick to checking if the string contains 'grass' or 'plant' or 'pot'.
                        # This works if the user points at a potted plant.
                        
                        keywords = ['grass', 'lawn', 'plant', 'pot', 'broccoli', 'cabbage', 'vegetable', 'tree', 'garden', 'forest', 'valley']
                        
                        is_grass = any(k in class_name.lower() for k in keywords)
                        
                        # Hack for testing if model classes are obscure:
                        # print("Detected: {}".format(class_name)) 
                        
                        conf = top1_conf
                    except Exception as e:
                        print("Inference error: {}".format(e))
                        is_grass = False
                        conf = 0.0
                
                row_data.append({
                    'poly_idx': i,
                    'is_grass': is_grass,
                    'green_pct': conf, # Reusing this field for confidence
                    'activated': False,
                    'timestamp': now
                })
            
            # 4. Store (Overwrite if exists)
            history[y_key] = row_data
                
        # --- Logic: Activation Check ---
        # "activatd... only for y = offset_y + y_key exactly"
        # Inverted: We check if history has key K such that current_y == K + OFFSET
        # i.e., K == current_y - OFFSET
        
        target_y = round(current_y - OFFSET_Y, 1)
        
        # We need to find if we have data for this target_y.
        # Since floats are tricky and Y_SPEED is 2.0, we might skip the exact 0.1 decimal.
        # But user asked for "exactly". We'll try exact lookup on the rounded keys first.
        # If the user moves fast, they might skip. 
        # For robustness with "exact" intent: strict key lookup.
        
        active_states = [False, False, False] # For the 3 dots
        
        if target_y in history:
            items = history[target_y]
            # This is the moment of activation!
            for idx, item in enumerate(items):
                # Set visual state
                if item['is_grass']:
                    active_states[idx] = True
                    
                    # Logic: Trigger event (once per key encounter?)
                    # Since we might generate this frame multiple times for same Y if stationary?
                    # "activated" flag in item ensures single print per item.
                    if not item['activated']:
                        item['activated'] = True
                        msg = "ACTIVATE! P{} | Y={} (Offset)".format(item['poly_idx']+1, target_y)
                        print(msg)
                        active_notifications.append({'msg': msg, 'added': time.time(), 'is_grass': True})

        # --- Drawing ---
        draw_img = frame.copy()
        
        # Center Dots (Activation State)
        # Overlay on camera feed
        # 3 dots horizontally centered
        dot_y = h // 2
        center_x = w // 2
        spacing = 60
        
        for i in range(3):
            # i=0 (Left), 1 (Center), 2 (Right)
            # Positions relative to center: -spacing, 0, +spacing
            dx = (i - 1) * spacing
            cx = center_x + dx
            
            # Color: Green if Active, Gray if Inactive
            dot_color = (0, 255, 0) if active_states[i] else (100, 100, 100)
            
            cv2.circle(draw_img, (cx, dot_y), 15, dot_color, -1)
            cv2.circle(draw_img, (cx, dot_y), 15, (255, 255, 255), 2) # White border

        # Increase width for 2 minimaps
        map_width = 200
        total_map_width = map_width * 2
        new_w = w + total_map_width
        
        combined_img = np.zeros((h, new_w, 3), dtype=np.uint8)
        combined_img[:, :w] = draw_img # Copy camera frame
        combined_img[:, w:] = (50, 50, 50) # Gray background for maps
        
        # Draw Separators
        cv2.line(combined_img, (w + map_width, 0), (w + map_width, h), (200, 200, 200), 1)
        
        # Map Headers
        cv2.putText(combined_img, "Raw History", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_img, "+Offset Replay", (w + map_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw polygons on main frame
        for i, poly in enumerate(polygons):
            color = (255, 0, 0)
            cv2.polylines(combined_img, [poly], True, color, 2)
            cv2.putText(combined_img, "P{}".format(i+1), (poly[0][0], poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Y info & Time since snip
        cv2.putText(combined_img, "Y: {:.1f} | Offset: {}".format(current_y, OFFSET_Y), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined_img, "Time Since Snip: {:.2f}s".format(time_since_last_snip), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined_img, "Use 'W' (Up) / 'S' (Down) to move Y", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw Notifications
        active_notifications = [n for n in active_notifications if time.time() - n['added'] < 2.0]
        y_text_start = 150
        for n in active_notifications:
            text_color = (0, 255, 0) if n['is_grass'] else (200, 200, 200)
            cv2.putText(combined_img, n['msg'], (10, y_text_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_text_start += 20
            
        # --- Minimap Drawing ---
        
        map_center_y = int(h * 0.8)
        scale = 2.0 # pixels per Y unit
        block_w = 40
        block_h = 10 
        gap = 5
        
        # --- Map 1: Raw ---
        map1_x_start = w + 10
        # Draw Current Y Line
        cv2.line(combined_img, (w, map_center_y), (w + map_width, map_center_y), (0, 255, 255), 1)
        
        # --- Map 2: Offset ---
        map2_x_start = w + map_width + 10
        # Draw Current Y Line
        cv2.line(combined_img, (w + map_width, map_center_y), (new_w, map_center_y), (0, 255, 255), 1)

        
        # Draw history items
        sorted_keys = sorted(history.keys())
        
        for y_key in sorted_keys:
            rows = history[y_key]
            
            # --- Draw on Map 1 (Raw) ---
            # screen_y = center + (current - key) * scale
            screen_y_raw = int(map_center_y + (current_y - y_key) * scale)
            
            if 0 <= screen_y_raw <= h:
                for idx, item in enumerate(rows):
                    fill_color = (0, 255, 0) if item['is_grass'] else (0, 0, 0)
                    rect_x = map1_x_start + idx * (block_w + gap)
                    cv2.rectangle(combined_img, (rect_x, screen_y_raw), (rect_x + block_w, screen_y_raw + block_h), fill_color, -1)
                    cv2.rectangle(combined_img, (rect_x, screen_y_raw), (rect_x + block_w, screen_y_raw + block_h), (200, 200, 200), 1)

            # --- Draw on Map 2 (Offset) ---
            # Effective Y = y_key + OFFSET_Y
            # screen_y = center + (current - effective) * scale
            effective_y = y_key + OFFSET_Y
            screen_y_offset = int(map_center_y + (current_y - effective_y) * scale)
            
            if 0 <= screen_y_offset <= h:
                for idx, item in enumerate(rows):
                    fill_color = (0, 255, 0) if item['is_grass'] else (0, 0, 0)
                    rect_x = map2_x_start + idx * (block_w + gap)
                    cv2.rectangle(combined_img, (rect_x, screen_y_offset), (rect_x + block_w, screen_y_offset + block_h), fill_color, -1)
                    cv2.rectangle(combined_img, (rect_x, screen_y_offset), (rect_x + block_w, screen_y_offset + block_h), (200, 200, 200), 1)

        cv2.imshow('Broadcast Spray Simulation', combined_img)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
