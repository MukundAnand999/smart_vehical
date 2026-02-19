import pygame
import random
import time
import json
import os
import threading
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Network
    BROKER = "broker.emqx.io"
    PORT = 1883
    TOPIC_VEHICLE = "v2x/hackathon/final/cars"
    TOPIC_EMERGENCY = "v2x/hackathon/final/emergency"

    # Window
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    FPS = 60

    # Road
    LANE_HEIGHT = 120
    LANE_COUNT = 4 # 3 Main + 1 Service
    SERVICE_LANE_INDEX = 3
    ROAD_Y = 50
    TOTAL_LENGTH = 2000 

    # Physics
    MAX_SPEED_KMH = 250
    MAX_SPEED_MS = MAX_SPEED_KMH / 3.6 
    ACCEL = 10   
    BRAKE = 25   
    LANE_CHANGE_SPEED = 2.0 
    
    # Safety
    SAFE_DIST = 200  
    CRITICAL_DIST = 80 
    AMBULANCE_DIST = 400 
    GHOST_TIMEOUT = 3.0 
    DROWSY_TIME_THRESHOLD = 2.0

    # Colors (Professional Palette)
    C_BG = (235, 240, 245)      # Soft Blue-Grey
    C_ROAD = (60, 60, 70)       # Dark Asphalt
    C_LINE = (220, 220, 220)    # Soft White
    C_LINE_YELLOW = (240, 200, 50) # Muted Yellow
    C_PANEL = (40, 44, 50)      # Dark Matte Panel
    C_TEXT = (230, 230, 230)    # Off-White Text
    C_WARNING = (255, 60, 60)   # Alert Red
    C_SERVICE_TEXT = (100, 100, 110)

# ==========================================
# 2. CAMERA DETECTOR (Fixed Rotation)
# ==========================================
class UniversalDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.drowsy = False
        self.no_face_start_time = None
        self.running = True
        self.current_frame = None 
        self.lock = threading.Lock()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        print("✅ Camera Active")

    def loop(self):
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: continue

            # Detection Logic (Grayscale)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                if self.no_face_start_time is None:
                    self.no_face_start_time = time.time()
                elif time.time() - self.no_face_start_time > Config.DROWSY_TIME_THRESHOLD:
                    self.drowsy = True
            else:
                self.no_face_start_time = None
                self.drowsy = False
            
            # Draw Face Box
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            with self.lock:
                # 1. Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 2. Rotate 90 degrees (Correcting for Pygame axis)
                self.current_frame = np.rot90(rgb_frame)

            time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                surf = pygame.surfarray.make_surface(self.current_frame)
                # 3. Flip Horizontal (Mirror Effect)
                return pygame.transform.flip(surf, True, False)
        return None

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

# ==========================================
# 3. HELPER CLASSES
# ==========================================
class Button:
    def __init__(self, x, y, w, h, text, color=(0, 100, 200), cb=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.cb = cb
        self.hover = False

    def draw(self, screen, font):
        # Subtle hover effect
        col = (min(self.color[0]+20,255), min(self.color[1]+20,255), min(self.color[2]+20,255)) if self.hover else self.color
        pygame.draw.rect(screen, col, self.rect, border_radius=6)
        pygame.draw.rect(screen, (150,150,150), self.rect, 1, border_radius=6)
        txt = font.render(self.text, True, (255,255,255))
        screen.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

    def check_click(self, pos):
        if self.rect.collidepoint(pos) and self.cb:
            self.cb()

class Vehicle:
    def __init__(self, uid, is_emergency=False):
        self.id = uid
        self.is_emergency = is_emergency
        self.lane = 1 if not is_emergency else 0
        self.visual_lane = float(self.lane)
        self.x = 0 if is_emergency else random.randint(0, 1500)
        self.speed = Config.MAX_SPEED_MS if is_emergency else random.randint(20, 30)
        self.user_target_speed = 25 
        self.target_speed = self.speed
        self.color = (255, 215, 0) if is_emergency else (random.randint(50, 200), random.randint(50, 200), 255)
        self.warning_vehicle_ahead = False 
        self.braking = False
        self.drowsy_alert = False
        self.last_update = time.time()
        self.lane_change_cooldown = 0

    def update_physics(self, dt, all_vehicles, is_me_drowsy=False):
        dist_to_ahead = float('inf')
        speed_of_car_ahead = None
        ambulance_behind = False
        
        for other in all_vehicles.values():
            if other.id == self.id: continue
            d_ahead = other.x - self.x
            d_behind = self.x - other.x
            if d_ahead < 0: d_ahead += Config.TOTAL_LENGTH
            if d_behind < 0: d_behind += Config.TOTAL_LENGTH

            if other.lane == self.lane:
                if d_ahead < dist_to_ahead:
                    dist_to_ahead = d_ahead
                    speed_of_car_ahead = other.speed
                if other.is_emergency and d_behind < Config.AMBULANCE_DIST:
                    ambulance_behind = True

        self.warning_vehicle_ahead = False
        self.braking = False
        self.lane_change_cooldown -= dt

        # === LOGIC ===
        if is_me_drowsy:
            self.drowsy_alert = True
            if self.lane != Config.SERVICE_LANE_INDEX:
                if self.lane_change_cooldown <= 0:
                    self.lane += 1
                    self.lane_change_cooldown = 1.0
            else:
                self.target_speed = 0
                self.braking = True
        
        elif ambulance_behind and not self.is_emergency and self.lane_change_cooldown <= 0:
            if self.lane < Config.LANE_COUNT - 1: self.lane += 1; self.lane_change_cooldown = 2.0
            elif self.lane > 0: self.lane -= 1; self.lane_change_cooldown = 2.0

        elif self.is_emergency:
            self.target_speed = Config.MAX_SPEED_MS * 1.2
        elif dist_to_ahead < Config.CRITICAL_DIST:
            self.target_speed = 0
            self.warning_vehicle_ahead = True
            self.braking = True
        elif dist_to_ahead < Config.SAFE_DIST:
            if speed_of_car_ahead is not None:
                self.target_speed = speed_of_car_ahead * 0.8 
            else:
                self.target_speed = 10 
            self.braking = True 
            if self.speed > (speed_of_car_ahead or 0): self.warning_vehicle_ahead = True
        else:
            self.target_speed = self.user_target_speed
        
        # Physics
        if self.speed < self.target_speed: self.speed += Config.ACCEL * dt
        elif self.speed > self.target_speed: self.speed -= Config.BRAKE * dt

        self.speed = max(0, min(self.speed, Config.MAX_SPEED_MS * 1.5))
        self.x += self.speed * dt
        if self.x > Config.TOTAL_LENGTH: self.x = 0

    def update_visuals(self, dt):
        diff = self.lane - self.visual_lane
        if abs(diff) > 0.01:
            direction = 1 if diff > 0 else -1
            self.visual_lane += direction * Config.LANE_CHANGE_SPEED * dt
            if abs(self.lane - self.visual_lane) < 0.05:
                self.visual_lane = float(self.lane)
        else:
            self.visual_lane = float(self.lane)

    def to_json(self):
        return json.dumps({
            "id": self.id, "lane": self.lane, "x": self.x, 
            "spd": self.speed, "emb": self.is_emergency, 
            "col": self.color, "drw": self.drowsy_alert, "brk": self.braking
        })

    @staticmethod
    def from_json(payload):
        try:
            d = json.loads(payload)
            v = Vehicle(d['id'], d['emb'])
            v.lane = d['lane']; v.x = d['x']; v.speed = d['spd']; v.color = tuple(d['col'])
            v.drowsy_alert = d.get('drw', False)
            v.braking = d.get('brk', False)
            v.visual_lane = float(v.lane)
            v.last_update = time.time()
            return v
        except: return None

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
class V2XApp:
    def __init__(self, uid):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption(f"V2X Professional - {uid}")
        self.clock = pygame.time.Clock()
        
        # PROFESSIONAL FONTS (Reduced Size)
        try:
            self.font = pygame.font.SysFont("Verdana", 18)        # Reduced from 24
            self.big_font = pygame.font.SysFont("Verdana", 28, bold=True) # Reduced from 36
            self.road_font = pygame.font.SysFont("Arial Narrow", 50, bold=True)
            self.alert_font = pygame.font.SysFont("Verdana", 36, bold=True)
        except:
            self.font = pygame.font.Font(None, 24)
            self.big_font = pygame.font.Font(None, 32)

        self.my_vehicle = Vehicle(uid)
        self.my_vehicle.color = (0, 200, 255) 
        self.vehicles = {uid: self.my_vehicle}
        self.i_own_ambulance = False 
        self.images = self.load_assets()
        
        self.manual_drowsy = False
        self.any_drowsy_detected = False

        print("Initializing Camera...")
        self.detector = UniversalDetector()

        # UI CONTROLS (Moved down slightly to fit panel)
        y = 680
        btn_w = 110
        self.buttons = [
            Button(50, y, btn_w, 40, "< Left", cb=lambda: self.change_lane(-1)),
            Button(170, y, btn_w, 40, "Right >", cb=lambda: self.change_lane(1)),
            Button(400, y, btn_w, 40, "Slower", color=(180,60,60), cb=lambda: self.chg_speed(-5)),
            Button(520, y, btn_w, 40, "Faster", color=(60,180,60), cb=lambda: self.chg_speed(5)),
            Button(800, y, 220, 50, "CALL AMBULANCE", color=(220,40,40), cb=self.spawn_ambulance_click)
        ]

        self.client = mqtt.Client(client_id=f"{uid}_{random.randint(1000,9999)}", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self.on_message
        self.client.connect(Config.BROKER, Config.PORT)
        self.client.subscribe(Config.TOPIC_VEHICLE)
        self.client.loop_start()

    def load_assets(self):
        imgs = {}
        CAR_SIZE = (110, 55) # Slightly smaller cars
        try:
            if os.path.exists("assets/car_blue.png"):
                imgs['me'] = pygame.transform.scale(pygame.image.load("assets/car_blue.png"), CAR_SIZE)
                imgs['other'] = pygame.transform.scale(pygame.image.load("assets/car_red.png"), CAR_SIZE)
                imgs['amb'] = pygame.transform.scale(pygame.image.load("assets/ambulance.png"), CAR_SIZE)
        except: pass
        return imgs

    def change_lane(self, d):
        new_lane = self.my_vehicle.lane + d
        if 0 <= new_lane < Config.LANE_COUNT: self.my_vehicle.lane = new_lane

    def chg_speed(self, d):
        self.my_vehicle.user_target_speed += d
        self.my_vehicle.user_target_speed = max(0, min(self.my_vehicle.user_target_speed, Config.MAX_SPEED_MS))

    def spawn_ambulance_click(self):
        if "AMB-1" not in self.vehicles:
            amb = Vehicle("AMB-1", is_emergency=True)
            self.vehicles["AMB-1"] = amb
            self.i_own_ambulance = True

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            d = json.loads(payload)
            if d['id'] != self.my_vehicle.id:
                if d['id'] == "AMB-1" and self.i_own_ambulance: return
                if d['id'] not in self.vehicles: 
                    self.vehicles[d['id']] = Vehicle.from_json(payload)
                else:
                    v = self.vehicles[d['id']]
                    v.x = d['x']; v.lane = d['lane']; v.speed = d['spd']; v.is_emergency = d['emb']
                    v.drowsy_alert = d.get('drw', False)
                    v.braking = d.get('brk', False)
                    v.last_update = time.time()
        except: pass

    def cleanup_ghosts(self):
        now = time.time()
        ghosts = [vid for vid, v in self.vehicles.items() 
                  if vid != self.my_vehicle.id 
                  and (now - v.last_update > Config.GHOST_TIMEOUT)]
        if self.i_own_ambulance and "AMB-1" in ghosts: ghosts.remove("AMB-1")
        for vid in ghosts: del self.vehicles[vid]

    def draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_len=20):
        x1, y1 = start_pos
        x2, y2 = end_pos
        for x in range(x1, x2, dash_len * 2):
            pygame.draw.line(surface, color, (x, y1), (min(x + dash_len, x2), y1), width)

    def draw_car(self, v):
        x = (v.x / Config.TOTAL_LENGTH) * Config.SCREEN_WIDTH
        y = Config.ROAD_Y + (v.visual_lane * Config.LANE_HEIGHT) + 35
        
        drawn = False
        if self.images:
            if v.is_emergency and 'amb' in self.images:
                self.screen.blit(self.images['amb'], (x, y)); drawn = True
            elif v.id == self.my_vehicle.id and 'me' in self.images:
                self.screen.blit(self.images['me'], (x, y)); drawn = True
            elif 'other' in self.images:
                self.screen.blit(self.images['other'], (x, y)); drawn = True

        if not drawn:
            col = v.color
            if v.is_emergency and (pygame.time.get_ticks() % 200 < 100): col = (255, 50, 50)
            pygame.draw.rect(self.screen, col, (x, y, 110, 55), border_radius=10)
            pygame.draw.rect(self.screen, (0,0,0), (x, y, 110, 55), 2, border_radius=10)
            pygame.draw.rect(self.screen, (20,20,40), (x+70, y+5, 30, 45), border_radius=6)
        
        brake_col = (255, 0, 0) if v.braking else (100, 0, 0)
        pygame.draw.circle(self.screen, brake_col, (x+5, y+12), 6)
        pygame.draw.circle(self.screen, brake_col, (x+5, y+43), 6)
        txt = self.font.render(v.id, True, (255,255,255) if not drawn else (0,0,0))
        self.screen.blit(txt, (x, y-25))
        
        if v.drowsy_alert:
             zzz = self.big_font.render("ZZZ", True, (255, 0, 255))
             self.screen.blit(zzz, (x + 30, y - 50))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(Config.FPS) / 1000.0
            mx, my = pygame.mouse.get_pos()
            blink_on = pygame.time.get_ticks() % 600 < 300 

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    self.manual_drowsy = not self.manual_drowsy
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for b in self.buttons: b.check_click((mx, my))
            for b in self.buttons: b.hover = b.rect.collidepoint((mx, my))

            # Update
            is_drowsy = self.detector.drowsy or self.manual_drowsy
            self.my_vehicle.update_physics(dt, self.vehicles, is_me_drowsy=is_drowsy)
            
            self.any_drowsy_detected = is_drowsy
            for v in self.vehicles.values():
                if v.drowsy_alert: self.any_drowsy_detected = True

            if self.i_own_ambulance and "AMB-1" in self.vehicles:
                amb = self.vehicles["AMB-1"]
                amb.update_physics(dt, self.vehicles)
                self.client.publish(Config.TOPIC_VEHICLE, amb.to_json())

            for v in self.vehicles.values(): v.update_visuals(dt)
            self.client.publish(Config.TOPIC_VEHICLE, self.my_vehicle.to_json())
            self.cleanup_ghosts()

            # Draw
            self.screen.fill(Config.C_BG)
            road_h = Config.LANE_HEIGHT * Config.LANE_COUNT
            pygame.draw.rect(self.screen, Config.C_ROAD, (0, Config.ROAD_Y, Config.SCREEN_WIDTH, road_h))
            
            for i in range(Config.LANE_COUNT+1):
                y = Config.ROAD_Y + i*Config.LANE_HEIGHT
                line_color = Config.C_LINE
                if i == Config.SERVICE_LANE_INDEX:
                    line_color = Config.C_LINE_YELLOW 
                    pygame.draw.line(self.screen, line_color, (0,y), (Config.SCREEN_WIDTH,y), 4)
                elif i == 0 or i == Config.LANE_COUNT:
                    pygame.draw.line(self.screen, line_color, (0,y), (Config.SCREEN_WIDTH,y), 3)
                else:
                    self.draw_dashed_line(self.screen, line_color, (0, y), (Config.SCREEN_WIDTH, y), 3, 30)

            svc_y = Config.ROAD_Y + (Config.SERVICE_LANE_INDEX * Config.LANE_HEIGHT) + 40
            svc_txt = self.road_font.render("SERVICE LANE", True, Config.C_SERVICE_TEXT)
            for x in range(200, Config.SCREEN_WIDTH, 600):
                self.screen.blit(svc_txt, (x, svc_y))

            for v in self.vehicles.values(): self.draw_car(v)

            # --- BOTTOM PANEL ---
            pygame.draw.rect(self.screen, Config.C_PANEL, (0, 640, Config.SCREEN_WIDTH, 160))
            for b in self.buttons: b.draw(self.screen, self.font)
            
            spd_kmh = int(self.my_vehicle.speed * 3.6)
            self.screen.blit(self.big_font.render(f"Speed: {spd_kmh} km/h", True, Config.C_TEXT), (50, 750))

            # --- ALERTS AREA (Center) ---
            
            # 1. BRAKING ALERT (My Car)
            if self.my_vehicle.braking and blink_on:
                alert = self.alert_font.render("BRAKING", True, (255, 100, 100))
                self.screen.blit(alert, (Config.SCREEN_WIDTH//2 - alert.get_width()//2, 650))

            # 2. PROXIMITY ALERT (Box in bottom right)
            if self.my_vehicle.warning_vehicle_ahead:
                box_rect = pygame.Rect(Config.SCREEN_WIDTH - 280, 650, 260, 60)
                pygame.draw.rect(self.screen, (120, 0, 0), box_rect, border_radius=10)
                pygame.draw.rect(self.screen, (255, 0, 0), box_rect, 2, border_radius=10)
                txt = self.font.render("VEHICLE AHEAD!", True, (255, 255, 255))
                self.screen.blit(txt, (box_rect.x + 40, box_rect.y + 18))

            # 3. GLOBAL DROWSINESS ALERT (Top Center)
            if self.any_drowsy_detected and blink_on:
                warn_surf = self.alert_font.render("⚠️ DROWSY DRIVER DETECTED ⚠️", True, (255, 255, 255))
                pygame.draw.rect(self.screen, (220, 0, 0), (Config.SCREEN_WIDTH//2 - 420, 5, 840, 50), border_radius=10)
                self.screen.blit(warn_surf, (Config.SCREEN_WIDTH//2 - warn_surf.get_width()//2, 10))

            # --- CAMERA FEED (Top Right - HUD Style) ---
            # Moves to top right so it doesn't overlap buttons
            frame_surf = self.detector.get_frame()
            if frame_surf:
                frame_surf = pygame.transform.scale(frame_surf, (240, 180))
                self.screen.blit(frame_surf, (Config.SCREEN_WIDTH - 260, 60))
                pygame.draw.rect(self.screen, (0, 255, 255), (Config.SCREEN_WIDTH - 260, 60, 240, 180), 2)
                cam_txt = self.font.render("DRIVER CAM", True, (0, 255, 255))
                self.screen.blit(cam_txt, (Config.SCREEN_WIDTH - 255, 65))

            pygame.display.flip()

        self.detector.stop()
        pygame.quit()
        self.client.loop_stop()

if __name__ == "__main__":
    uid = input("Enter Vehicle ID (e.g. CAR1): ").strip().upper() or "CAR1"
    app = V2XApp(uid)
    app.run()