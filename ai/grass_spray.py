import cv2
import numpy as np
import time

def detect_grass_mask(img, config, prev_mask=None):
    # 1. Illumination Normalization
    img_float = img.astype(np.float32) + 1e-6
    # Sum across channels (B+G+R)
    total = img_float.sum(axis=2, keepdims=True)
    norm = img_float / total
    
    # Split normalized channels. Order is BGR
    b, g, r = cv2.split(norm)

    # 2. Vegetation index on normalized space (ExG)
    # exg = 2*g - r - b
    exg = 2*g - r - b
    # Normalize to 0-255 uint8
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 3. Adaptive threshold
    mean_val = np.mean(exg)
    std_val = np.std(exg)
    # Configurable Std Factor
    thr = mean_val + config['exg_std_factor'] * std_val
    mask = (exg > thr).astype("uint8") * 255

    # 4. Spatial coherence
    # a) Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # b) Connected components filter (Remove tiny false positives)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    # Configurable Min/Max Area
    min_area = config['min_area_pct'] * img.shape[0] * img.shape[1]
    max_area = config['max_area_pct'] * img.shape[0] * img.shape[1]

    clean = np.zeros_like(mask)
    # Skip label 0 (background)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area and area < max_area:
            clean[labels == i] = 255
            
    # 5. Texture confirmation (Variance Method)
    # Canny detects edges, but Variance detects "roughness" (better for grass vs smooth paper)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # K-size determines the scale of texture checking
    ksize = (9, 9)
    mu = cv2.blur(gray, ksize)
    mu2 = cv2.blur(gray*gray, ksize)
    
    # Variance = E[X^2] - (E[X])^2
    variance = mu2 - mu**2
    std_dev = np.sqrt(np.maximum(variance, 0))
    
    # Threshold for "texturedness"
    texture_mask = (std_dev > config['texture_std_thr']).astype(np.uint8) * 255
    
    # Enforce that grass regions must have ExG Greenness AND High Texture
    final_mask = cv2.bitwise_and(clean, texture_mask)

    # 6. Temporal stabilization
    alpha = config['temporal_alpha']
    if prev_mask is not None:
        # alpha is for previous frame weight?
        # Typically addWeighted(src1, alpha, src2, beta, gamma)
        # If user sets Alpha=0.7, assume they want 0.7 persistence of old mask
        final_mask = cv2.addWeighted(prev_mask, alpha, final_mask, 1.0 - alpha, 0)
        
    return final_mask, clean, texture_mask

def on_trackbar(val):
    pass

def main():
    cap = cv2.VideoCapture(0)
    
    # --- Configuration Defaults ---
    config = {
        'exg_std_factor': 0.25,
        'min_area_pct': 0.001, # Lower default min area
        'max_area_pct': 0.30,  # Ignore huge blobs (walls/tables) > 30% screen
        'texture_std_thr': 25.0, # Increased default texture threshold
        'temporal_alpha': 0.5,   # Less ghosting
        'green_threshold': 0.1
    }
    
    # Setup Window & Trackbars
    window_name = 'Broadcast Spray Simulation'
    cv2.namedWindow(window_name)
    
    cv2.createTrackbar('ExG Std x100', window_name, int(config['exg_std_factor']*100), 400, on_trackbar)
    cv2.createTrackbar('Min Area % x1k', window_name, int(config['min_area_pct']*1000), 100, on_trackbar)
    cv2.createTrackbar('Max Area % x100', window_name, int(config['max_area_pct']*100), 100, on_trackbar)
    cv2.createTrackbar('Tex Std x10', window_name, int(config['texture_std_thr']*10), 1000, on_trackbar) # Increased range
    cv2.createTrackbar('Temp Alpha x10', window_name, int(config['temporal_alpha']*10), 10, on_trackbar)
    # cv2.createTrackbar('Green Thr x100', window_name, int(config['green_threshold']*100), 100, on_trackbar)
    
    # Polygons (Frames). defined as list of (x, y) points.
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
    history = {}
    last_snapshot_time = time.time()
    last_snapshot_y = -999.0
    prev_mask = None # For temporal stabilization
    
    # For display of activation
    active_notifications = [] # list of messages to show
    
    print("Controls:")
    print("  UP/DOWN ('W'/'S'): Move Y variable")
    print("  Adjust Trackbars to tune Detection")
    print("  'q' or 'ESC': Quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Flip for user friendliness
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- Update Config from Trackbars ---
        # Get current positions
        # --- Update Config from Trackbars ---
        # Get current positions
        config['exg_std_factor'] = cv2.getTrackbarPos('ExG Std x100', window_name) / 100.0
        config['min_area_pct'] = cv2.getTrackbarPos('Min Area % x1k', window_name) / 1000.0
        config['max_area_pct'] = cv2.getTrackbarPos('Max Area % x100', window_name) / 100.0
        if config['max_area_pct'] == 0: config['max_area_pct'] = 1.0 # Safety
        
        config['texture_std_thr'] = cv2.getTrackbarPos('Tex Std x10', window_name) / 10.0
        config['temporal_alpha'] = cv2.getTrackbarPos('Temp Alpha x10', window_name) / 10.0
        # config['green_threshold'] = cv2.getTrackbarPos('Green Thr x100', window_name) / 100.0

        # --- Input Handling ---
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'): # ESC or q
            break
        elif key == 0 or key == 2490368: # Up arrow (platform dependent sometimes)
            pass
            
        if cv2.waitKey(1) == -1: pass
        
        if key == ord('w'): # Up
            current_y += Y_SPEED
        elif key == ord('s'): # Down
            current_y -= Y_SPEED
            
        # --- Logic: Run Pipeline Per Frame (For Temporal Stability + Viz) ---
        grass_mask, debug_green, debug_texture = detect_grass_mask(frame, config, prev_mask)
        prev_mask = grass_mask # Update for next frame
        
        # --- Visualization Overlays ---
        # Scale down for debug view
        debug_scale = 0.25
        d_w = int(w * debug_scale)
        d_h = int(h * debug_scale)
        
        # 1. Green Mask (Visualized as Green)
        green_vis = cv2.resize(debug_green, (d_w, d_h))
        green_vis_bgr = cv2.cvtColor(green_vis, cv2.COLOR_GRAY2BGR)
        green_vis_bgr[:, :, 0] = 0 # B
        green_vis_bgr[:, :, 2] = 0 # R
        
        # 2. Texture Mask (Visualized as Blue)
        tex_vis = cv2.resize(debug_texture, (d_w, d_h))
        tex_vis_bgr = cv2.cvtColor(tex_vis, cv2.COLOR_GRAY2BGR)
        tex_vis_bgr[:, :, 1] = 0 # G
        tex_vis_bgr[:, :, 2] = 0 # R
        
        # Overlay Logic
        # Top-Left: Green
        frame[0:d_h, 0:d_w] = cv2.addWeighted(frame[0:d_h, 0:d_w], 0.5, green_vis_bgr, 0.5, 0)
        cv2.putText(frame, "Green (ExG)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Top-Left + Offset: Texture
        frame[0:d_h, d_w:d_w*2] = cv2.addWeighted(frame[0:d_h, d_w:d_w*2], 0.5, tex_vis_bgr, 0.5, 0)
        cv2.putText(frame, "Texture (Var)", (d_w+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # --- Logic: Snapshot ---
        now = time.time()
        time_since_last_snip = now - last_snapshot_time
        y_change = abs(current_y - last_snapshot_y)
        
        if time_since_last_snip > SNAPSHOT_INTERVAL or y_change >= 1.0:
            last_snapshot_time = now
            last_snapshot_y = current_y
            
            # Use computed grass_mask
            
            # Prepare row data for this Y
            y_key = round(current_y, 1)
            row_data = [] # List of 3 polygon states
            
            for i, poly in enumerate(polygons):
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [poly], 255)
                
                region_grass_val = cv2.bitwise_and(grass_mask, grass_mask, mask=poly_mask)
                
                # BUG FIX: Threshold the temporally smoothed mask.
                # Threshold at 50/255 (approx 20% intensity)
                _, region_grass_thresh = cv2.threshold(region_grass_val, 50, 255, cv2.THRESH_BINARY)
                
                total_pixels = cv2.countNonZero(poly_mask)
                grass_pixels = cv2.countNonZero(region_grass_thresh)
                
                detect_ratio = 0.0
                if total_pixels > 0:
                    detect_ratio = grass_pixels / total_pixels
                
                # Use Configurable Threshold
                is_grass = detect_ratio > config['green_threshold']
                conf = detect_ratio 
                
                row_data.append({
                    'poly_idx': i,
                    'is_grass': is_grass,
                    'green_pct': conf, 
                    'activated': False,
                    'timestamp': now
                })
            
            history[y_key] = row_data
                
        # --- Logic: Activation Check ---
        target_y = round(current_y - OFFSET_Y, 1)
        
        active_states = [False, False, False]
        
        if target_y in history:
            items = history[target_y]
            for idx, item in enumerate(items):
                if item['is_grass']:
                    active_states[idx] = True
                    if not item['activated']:
                        item['activated'] = True
                        msg = "ACTIVATE! P{} | Y={} (Offset)".format(item['poly_idx']+1, target_y)
                        print(msg)
                        active_notifications.append({'msg': msg, 'added': time.time(), 'is_grass': True})

        # --- Drawing ---
        draw_img = frame.copy()
        
        # Center Dots
        dot_y = h // 2
        center_x = w // 2
        spacing = 60
        
        for i in range(3):
            dx = (i - 1) * spacing
            cx = center_x + dx
            dot_color = (0, 255, 0) if active_states[i] else (100, 100, 100)
            cv2.circle(draw_img, (cx, dot_y), 15, dot_color, -1)
            cv2.circle(draw_img, (cx, dot_y), 15, (255, 255, 255), 2)

        # Increase width for 2 minimaps
        map_width = 200
        total_map_width = map_width * 2
        new_w = w + total_map_width
        
        combined_img = np.zeros((h, new_w, 3), dtype=np.uint8)
        combined_img[:, :w] = draw_img
        combined_img[:, w:] = (50, 50, 50)
        
        cv2.line(combined_img, (w + map_width, 0), (w + map_width, h), (200, 200, 200), 1)
        
        cv2.putText(combined_img, "Raw History", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_img, "+Offset Replay", (w + map_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, poly in enumerate(polygons):
            color = (255, 0, 0)
            cv2.polylines(combined_img, [poly], True, color, 2)
            cv2.putText(combined_img, "P{}".format(i+1), (poly[0][0], poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(combined_img, "Y: {:.1f} | Offset: {}".format(current_y, OFFSET_Y), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined_img, "Time Since Snip: {:.2f}s".format(time_since_last_snip), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined_img, "Use 'W' (Up) / 'S' (Down) to move Y", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        active_notifications = [n for n in active_notifications if time.time() - n['added'] < 2.0]
        y_text_start = 150
        for n in active_notifications:
            text_color = (0, 255, 0) if n['is_grass'] else (200, 200, 200)
            cv2.putText(combined_img, n['msg'], (10, y_text_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_text_start += 20
            
        # --- Minimap Drawing ---
        map_center_y = int(h * 0.8)
        scale = 2.0 
        block_w = 40
        block_h = 10 
        gap = 5
        
        map1_x_start = w + 10
        cv2.line(combined_img, (w, map_center_y), (w + map_width, map_center_y), (0, 255, 255), 1)
        
        map2_x_start = w + map_width + 10
        cv2.line(combined_img, (w + map_width, map_center_y), (new_w, map_center_y), (0, 255, 255), 1)

        sorted_keys = sorted(history.keys())
        
        for y_key in sorted_keys:
            rows = history[y_key]
            
            screen_y_raw = int(map_center_y + (current_y - y_key) * scale)
            if 0 <= screen_y_raw <= h:
                for idx, item in enumerate(rows):
                    fill_color = (0, 255, 0) if item['is_grass'] else (0, 0, 0)
                    rect_x = map1_x_start + idx * (block_w + gap)
                    cv2.rectangle(combined_img, (rect_x, screen_y_raw), (rect_x + block_w, screen_y_raw + block_h), fill_color, -1)
                    cv2.rectangle(combined_img, (rect_x, screen_y_raw), (rect_x + block_w, screen_y_raw + block_h), (200, 200, 200), 1)

            effective_y = y_key + OFFSET_Y
            screen_y_offset = int(map_center_y + (current_y - effective_y) * scale)
            
            if 0 <= screen_y_offset <= h:
                for idx, item in enumerate(rows):
                    fill_color = (0, 255, 0) if item['is_grass'] else (0, 0, 0)
                    rect_x = map2_x_start + idx * (block_w + gap)
                    cv2.rectangle(combined_img, (rect_x, screen_y_offset), (rect_x + block_w, screen_y_offset + block_h), fill_color, -1)
                    cv2.rectangle(combined_img, (rect_x, screen_y_offset), (rect_x + block_w, screen_y_offset + block_h), (200, 200, 200), 1)

        cv2.imshow(window_name, combined_img)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
