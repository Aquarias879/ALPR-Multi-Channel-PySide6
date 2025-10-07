# -*- coding: utf-8 -*-
import os, sys, re, time, math, queue, threading, socket, requests, pyodbc, logging, faulthandler
from datetime import datetime,timedelta
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import yaml
import json
from logging.handlers import RotatingFileHandler

# --- PySide6 Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QWidget, 
                               QDialog, QGridLayout, QScrollArea, QFormLayout, QMessageBox, 
                               QHBoxLayout, QTreeWidget, QTreeWidgetItem, QGroupBox, QComboBox,
                               QLineEdit, QHeaderView, QFileDialog,QVBoxLayout,QStyleFactory)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QScreen
from qt_material import apply_stylesheet # <-- 1. Import the library
import sys

faulthandler.enable()

# ──────────────────────────────────────────────────────────────────────
# Configuration
NUM_CHANNELS   = 4
CHANNELS       = [str(i) for i in range(1, NUM_CHANNELS + 1)]
GRID_MAX_COLS  = 2
# ──────────────────────────────────────────────────────────────────────

# UI push for new log rows (date, time, plate, channel)
LOG_UI_QUEUE = queue.Queue(maxsize=200)

# Project helpers (use your paths)
from static.extra.license_detect import ObjectDetector as LicenseDetector
from static.extra.display_control import PacketCreator as DisplayController

# =============================================================================
#                        CONFIG / STATE
# =============================================================================
class AppConf:
    def __init__(self):
        # Concurrency / Caches
        self.CACHE_EXPIRATION_SECONDS = 600.0
        self.LOCK_TIMEOUT = 5.0
        self.saved_license_numbers = {}
        self.cache_lock = threading.RLock()
        self.detected_classes = []
        self.detected_classes_lock = threading.RLock()
        self.notin_json_lock = threading.RLock()
        self._door_status_lock = threading.RLock()
        self.door_status = {ch: False for ch in CHANNELS}

        # GUI / Streams
        self.rtsp_map    = {ch: '' for ch in CHANNELS}
        self.url_gates   = {ch: '' for ch in CHANNELS}
        self.license_addrs = {ch: '' for ch in CHANNELS}
        self.video_streams = []
        self.root = None  # Will hold the QMainWindow instance

        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_path = os.path.join(self.base_dir, "static", "icon", "cctv.ico")
        self.onnx_model1 = os.path.join(self.base_dir, "static", "models", "plate.onnx")
        self.onnx_model2 = os.path.join(self.base_dir, "static", "models", "best.onnx")
        self.fontpath    = os.path.join(self.base_dir, "static", "font", "SimSun.ttf")
        self.settings_yml = os.path.join(self.base_dir, "static", "settings.yml")

        # Logging
        self.log_dir = os.path.join(self.base_dir, "static")
        self.log_file = os.path.join(self.log_dir, "alpr.log")
        self._setup_logging()

        # Labels, Detectors, Regex
        self.class_names  = ['-']
        self.class_names2 = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
        self._init_detectors()
        self.license_pattern = re.compile(r"""
            ^(
                \d{1}[A-Z]{2}\d{3}$|\d{4,5}[A-Z]{1}\d{1}$|\d{1}[A-Z]{1}\d{4,5}$|
                [A-Z]{2}\d{2,3}$|[A-Z]{1}\d{4,5}$|[A-Z]{2}\d{3,5}$|[A-Z]{3}\d{3,5}$|
                [A-Z]\d[A-Z]\d{3,5}$|\d{3,4}[A-Z]{2}$|[A-Z]\d{2}\d{4}$|0[A-Z]\d{4}$|
                CD\d{4}$|\d{3}[A-Z]{3}$|\d{3}[A-Z]\d{1,2}$|\d{1}[A-Z]\d{4,5}$|\d{6,7}$
            )$""", re.VERBOSE)

    def _setup_logging(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(self.log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            logging.basicConfig(
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
                handlers=[file_handler, logging.StreamHandler(sys.stdout)]
            )
            logging.info("Logging initialized")
        except Exception as e:
            print(f"Logging setup error: {e}")

    def _init_detectors(self):
        try:
            self.plate_detector = LicenseDetector(self.onnx_model1, self.class_names,  conf_thres=0.3, iou_thres=0.7)
            self.text_detector  = LicenseDetector(self.onnx_model2, self.class_names2, conf_thres=0.3, iou_thres=0.7)
            logging.info("Detectors initialized")
        except Exception as e:
            logging.error(f"Detector init error: {e}", exc_info=True)
            raise

    def update_door_status(self, channel: str, status: bool):
        """
        Thread-safely updates the open/closed status for a specific door channel.
        """
        if not self._door_status_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning(f"Timeout acquiring lock to update door status for channel {channel}.")
            return
        try:
            self.door_status[channel] = status
            logging.debug(f"Door status for channel {channel} set to {status}.")
        except Exception as e:
            logging.error(f"Error updating door status for channel {channel}: {e}")
        finally:
            self._door_status_lock.release()

    def get_door_status(self, channel: str) -> bool:
        """
        Thread-safely retrieves the open/closed status of a door channel.
        Returns False if the lock cannot be acquired or the channel doesn't exist.
        """
        if not self._door_status_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning(f"Timeout acquiring lock to get door status for channel {channel}.")
            return False
        try:
            return self.door_status.get(channel, False)
        except Exception as e:
            logging.error(f"Error getting door status for channel {channel}: {e}")
            return False
        finally:
            self._door_status_lock.release()

    def add_detected_license(self, license_number: str) -> bool:
        """
        Thread-safely adds a license number to a temporary list of recent detections,
        ensuring no duplicates. Returns True if the license was newly added.
        """
        if not license_number:
            return False
        
        if not self.detected_classes_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning("Timeout acquiring lock to add detected license.")
            return False
        try:
            if license_number not in self.detected_classes:
                self.detected_classes.append(license_number)
                return True
            return False
        except Exception as e:
            logging.error(f"Error adding detected license '{license_number}': {e}")
            return False
        finally:
            self.detected_classes_lock.release()

    def clear_detected_licenses(self):
        """
        Thread-safely clears the list of recently detected licenses.
        """
        if not self.detected_classes_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning("Timeout acquiring lock to clear detected licenses.")
            return
        try:
            self.detected_classes.clear()
        except Exception as e:
            logging.error(f"Error clearing detected licenses: {e}")
        finally:
            self.detected_classes_lock.release()

    def get_last_detection_time(self, license_number: str) -> float:
        """
        Thread-safely gets the timestamp of the last time a license plate was detected.
        Returns 0.0 if not found or on error.
        """
        if not self.cache_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning("Timeout acquiring lock to get last detection time.")
            return 0.0
        try:
            return self.saved_license_numbers.get(license_number, 0.0)
        except Exception as e:
            logging.error(f"Error getting last detection time for '{license_number}': {e}")
            return 0.0
        finally:
            self.cache_lock.release()

    def update_saved_license(self, license_number: str, timestamp: float):
        """
        Thread-safely updates or adds a license number and its detection timestamp to the cache.
        """
        if not self.cache_lock.acquire(timeout=self.LOCK_TIMEOUT):
            logging.warning("Timeout acquiring lock to update saved license.")
            return
        try:
            self.saved_license_numbers[license_number] = timestamp
        except Exception as e:
            logging.error(f"Error updating saved license for '{license_number}': {e}")
        finally:
            self.cache_lock.release()

config = AppConf()

# =============================================================================
#                      STREAM SOURCE & DEVICE HELPERS
# =============================================================================
def normalize_stream_source(val):
    if val is None: return None
    if isinstance(val, int): return val
    s = str(val).strip()
    if s == "": return None
    if re.fullmatch(r"\d+", s):
        try: return int(s)
        except Exception: pass
    return s

def is_url_like(s) -> bool:
    return isinstance(s, str) and (s.startswith(("rtsp://","rtmp://","http://","https://")))

def packet_handle(text, color, device):
    try:
        parsed = urlparse(device); address, port = parsed.hostname, parsed.port
        if not address or not port: raise ValueError(f"Invalid device URL: {device}")
        packet = DisplayController(text, color_name=color).create_packet()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5); sock.connect((address, port)); sock.sendall(packet)
    except Exception as e: logging.error(f"packet_handle error: {e}")
# =============================================================================
#                        BACKEND LOGIC (controlManager)
# =============================================================================
class ThreadSafeCounter:
    def __init__(self): self.count = 0; self.lock = threading.Lock()
    def increment(self):
        with self.lock: self.count += 1; return self.count
    def reset(self):
        with self.lock: self.count = 0
counter = ThreadSafeCounter()

class controlManager:
    @staticmethod
    def MainControl(data, frame, license_address, url_gate, pos):
        """
        Main entry point for control logic, routing tasks based on channel type.
        """
        channel_type = data.get("type")
        if channel_type == "1":
            controlManager.handle_channel_one_task(frame, license_address, data, url_gate, pos)
        elif channel_type in CHANNELS: # Handles all other channels
            controlManager.controling_task(frame, license_address, data, url_gate, pos)

    @staticmethod
    def door_open(data, url_gate, channel):
        """
        Sends a request to open the gate and manages the door status with a cooldown.
        """
        try:
            if config.get_door_status(channel):
                logging.info(f"Gate for CH{channel} is already open (in cooldown).")
                return False
            
            config.update_door_status(channel, True)
            resp = requests.get(url_gate, timeout=5)
            resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            if resp.status_code == 200:
                logging.info(f"Gate for CH{channel} opened successfully.")
                # Start a timer to reset the door status after a delay
                threading.Timer(1.0, config.update_door_status, args=(channel, False)).start()
                return True
            else:
                logging.warning(f"Failed to open gate for CH{channel}, status code: {resp.status_code}")
                config.update_door_status(channel, False)
                return False
        except requests.RequestException as e:
            logging.error(f"Error sending open signal to CH{channel}: {e}")
            config.update_door_status(channel, False)
            return False

    @staticmethod
    def controling_task(frame, addr, data, url_gate, pos):
        """
        Handles logic for exit channels (2, 3, etc.).
        Checks if the vehicle is in the authorized exit list.
        """
        channel = data['type']
        license_plate = data['car_no']
        
        try:
            with controlManager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(1) FROM canout_List WHERE car_no = ?", (license_plate,))
                is_authorized = (cursor.fetchone() or [0])[0] == 1
                
                if is_authorized:
                    logging.info(f"Authorized vehicle '{license_plate}' detected at exit CH{channel}. Opening gate.")
                    controlManager.insert_data(data)
                    controlManager.save_carout(frame, data, pos)
                    controlManager.door_open(data, url_gate, channel)
                else:
                    logging.warning(f"Unauthorized vehicle '{license_plate}' detected at exit CH{channel}.")
                    if config.license_pattern.match(license_plate):
                        controlManager.save_notin(frame, data, pos)
        except Exception as e:
            logging.error(f"Error in controling_task for '{license_plate}': {e}", exc_info=True)

    @staticmethod
    def handle_channel_one_task(frame, addr, data, url_gate, pos):
        """
        Handles the more complex logic for the entrance channel (1).
        """
        license_plate = data['car_no']
        try:
            with controlManager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT type FROM canin_List WHERE car_no = ?", (license_plate,))
                db_result = cursor.fetchone()
                
                current_time = time.time()
                last_time = config.get_last_detection_time(license_plate)
                
                # Check if the vehicle is authorized and not in a cooldown period
                if db_result and (current_time - last_time > 5):
                    config.update_saved_license(license_plate, current_time)
                    packet_handle("辨識中\n請稍等", color='yellow', device=addr)
                    requests.get(url_gate, timeout=5).raise_for_status() # Trigger gate hardware
                    
                    vehicle_type = db_result[0]
                    type_text = '機車' if vehicle_type == 'M' else '汽車' if vehicle_type == 'C' else '未知'
                    
                    display_text = f"辨識成功\n{type_text}:{license_plate}"
                    controlManager.insert_data(data)
                    controlManager.save_carin(frame, data, pos)
                    packet_handle(display_text, color='green', device=addr)
                    
                    # Reset display after a delay
                    threading.Timer(2.0, packet_handle, args=("一車一桿減速慢行\n非住戶勿擅自進入", 'red', addr)).start()
                    counter.reset()
                    return

                # If detected again within cooldown, ignore
                elif db_result and (current_time - last_time <= 5):
                    return

                # If not in the main entrance list, save as "not in" and manage failed attempts
                if config.license_pattern.match(license_plate):
                    controlManager.save_notin(frame, data, pos)
                    if counter.increment() >= 5:
                        counter.reset()
                        packet_handle("辨識不成功\n請聯絡管理室", color='red', device=addr)
                        threading.Timer(2.0, packet_handle, args=("一車一桿減速慢行\n非住戶勿擅自進入", 'red', addr)).start()

        except Exception as e:
            logging.error(f"Error in handle_channel_one_task for '{license_plate}': {e}", exc_info=True)

    ### --- NEW/MODIFIED --- ###
    @staticmethod
    def query_vehicle_status(license_plate: str):
        """
        Queries the database to determine the status of a vehicle.
        Returns: A tuple (CODE, message_string)
        """
        if not license_plate:
            return ("INVALID", "無車牌")

        try:
            with controlManager.connection() as conn:
                cursor = conn.cursor()
                # 1. Check if it's in the allowed entry list
                cursor.execute("SELECT type FROM canin_List WHERE car_no = ?", (license_plate,))
                result = cursor.fetchone()
                if result:
                    v_type = "機車" if result[0] == 'M' else "汽車"
                    return ("CAN_IN", f"允許進入 ({v_type})")

                # 2. If not, check if it's already marked as inside
                cursor.execute("SELECT IsInside FROM cust0011 WHERE car_no = ?", (license_plate,))
                result = cursor.fetchone()
                if result and result[0] == 1:
                    return ("ALREADY_PARKED", "車輛已在場內")

                # 3. If neither of the above, it's not registered for entry
                return ("NOT_REGISTERED", "未註冊車輛")
        except Exception as e:
            logging.error(f"Error querying vehicle status for '{license_plate}': {e}")
            return ("DB_ERROR", "資料庫錯誤")

    @staticmethod
    def connection():
        connection_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "Server=192.168.2.102,1435;"
            "Database=PKM;"
            "UID=sa;"
            "PWD=$j53272162;"
            "MultipleActiveResultSets=True;"
            "Trusted_Connection=no;"
        )
        return pyodbc.connect(connection_str)

    @staticmethod
    def insert_data(data):
        """ Inserts a detection log into the database. """
        sql = "INSERT INTO nlog0001 (car_no, type, add_date, add_time) VALUES (?, ?, ?, ?)"
        values = (data['car_no'], data['type'], data['add_date'], data['add_time'])
        try:
            with controlManager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, values)
                conn.commit()
                # Push the new log to the UI queue
                LOG_UI_QUEUE.put_nowait(values)
        except queue.Full:
            logging.warning("UI log queue is full, dropping new log entry.")
        except Exception as e:
            logging.error(f"Database insert error: {e}")
    
    @staticmethod
    def query_recent_logs(limit=100):
        rows = []
        try:
            with controlManager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT TOP {int(limit)} add_date, add_time, car_no, type FROM nlog0001 ORDER BY add_date DESC, add_time DESC")
                rows = [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in cursor.fetchall()]
        except Exception as e:
            logging.error(f"query_recent_logs error: {e}")
        return rows
    
    @staticmethod
    def delete_oldest_files(directory, max_files=500, delete_batch_size=100):
        """ Keeps the number of files in a directory under a specified limit. """
        try:
            files = sorted([os.path.join(directory, f) for f in os.listdir(directory)], key=os.path.getctime)
            if len(files) > max_files:
                for f in files[:delete_batch_size]:
                    try: os.remove(f)
                    except OSError: pass
        except Exception as e:
            logging.error(f"Error deleting old files in {directory}: {e}")

    @staticmethod
    def save_notin(frame, data, pos):
        """ Saves an image of an unrecognized vehicle. """
        license_plate = data['car_no']
        if (time.time() - config.get_last_detection_time(license_plate)) < config.CACHE_EXPIRATION_SECONDS:
            return
        config.update_saved_license(license_plate, time.time())
        save_dir = os.path.join(config.base_dir, 'static', 'img', 'notin_car')
        os.makedirs(save_dir, exist_ok=True)
        controlManager.delete_oldest_files(save_dir)
        path = os.path.join(save_dir, f"{data['car_no']}_{data['type']}_{data['add_date']}_{data['add_time']}.jpg")
        cv2.imwrite(path, frame)

    @staticmethod
    def save_carin(frame, data, pos):
        """ Saves an image of a vehicle entering. """
        save_dir = os.path.join(config.base_dir, 'static', 'img', 'car_in')
        os.makedirs(save_dir, exist_ok=True)
        controlManager.delete_oldest_files(save_dir)
        path = os.path.join(save_dir, f"{data['car_no']}_{data['type']}_{data['add_date']}_{data['add_time']}.jpg")
        cv2.imwrite(path, frame)

    @staticmethod
    def save_carout(frame, data, pos):
        """ Saves an image of a vehicle exiting. """
        save_dir = os.path.join(config.base_dir, 'static', 'img', 'car_out')
        os.makedirs(save_dir, exist_ok=True)
        controlManager.delete_oldest_files(save_dir)
        path = os.path.join(save_dir, f"{data['car_no']}_{data['type']}_{data['add_date']}_{data['add_time']}.jpg")
        cv2.imwrite(path, frame)

# =============================================================================
#                      VIDEO CAPTURE WORKER (Adapted for PySide)
# =============================================================================
class VideoCapture:
    def __init__(self, rtsp_url, label_widget: QLabel,main_window, **kwargs):
        self.rtsp_url = rtsp_url
        self.label = label_widget
        self.main_window = main_window
        self.q = queue.Queue(maxsize=2)
        self.running = True
        self.ch = kwargs.get('ch')
        self.executor = kwargs.get('executor')
        self.thread = threading.Thread(target=self._reader, daemon=True, name=f"RTSP-{self.ch}")
        self.thread.start()

    def _reader(self):
        while self.running:
            cap = None
            try:
                source = normalize_stream_source(self.rtsp_url)
                cap = cv2.VideoCapture(source)
                # In VideoCapture._reader
                if not cap.isOpened():
                    # It now calls the thread-safe method in MainApplication via a QTimer
                    self.update_label_error(f"Cannot connect to\n{self.rtsp_url}")
                    time.sleep(5)
                    continue
                
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if not self.q.full(): self.q.put_nowait(frame)
            except Exception as e:
                logging.error(f"Reader loop error for {self.rtsp_url}: {e}")
                time.sleep(5)
            finally:
                if cap: cap.release()

    def read(self):
        try: return self.q.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join(timeout=2)
            
    def update_label_error(self, message: str):
        try:
            pixmap = QPixmap(self.label.size())
            pixmap.fill(Qt.GlobalColor.red)
            painter = QPainter(pixmap)
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("SimSun", max(12, self.label.height() // 15)))
            painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.TextWordWrap, message)
            painter.end()
            self.label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"update_label_error failed: {e}")

    def show_frame_on_label(self, frame_bgr):
        try:
            if not self.label: return
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logging.error(f"show_frame_on_label failed: {e}", exc_info=True)

    def detect_objects(self, frame):
        # This is a direct port of your original detection logic
        try:
            h, w, _ = frame.shape
            roi_x, roi_y, roi_w, roi_h = 0, h // 4, w, h * 3 // 4
            cropped = frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

            boxes, _, _ = config.plate_detector(cropped)
            for box in boxes:
                x, y, w2, h2 = box
                x1, y1 = max(0, int(x) + roi_x), max(0, int(y) + roi_y)
                x2, y2 = min(w, int(w2) + roi_x), min(h, int(h2) + roi_y)
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0: continue

                resized = cv2.resize(plate_img, (256, 256))
                b2, _, ids2 = config.text_detector(resized)
                seq = sorted(zip(b2, ids2), key=lambda t: t[0][0])
                license_number = ''.join(config.class_names2[cid] for _, cid in seq)

                if license_number and config.add_detected_license(license_number):
                    now = datetime.now()
                    data = {"car_no": license_number, "type": self.ch, "add_date": f"{now.year-1911}{now:%m%d}", "add_time": f"{now:%H%M%S}"}
                    cv2.putText(frame, license_number, (roi_x, roi_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 2)
                    self.executor.submit(controlManager.MainControl, data, frame, config.license_addrs.get(self.ch), config.url_gates.get(self.ch), {"x":x,"y":y})
                    self.executor.submit(self.main_window.check_and_update_status, license_number)
                    
                    config.clear_detected_licenses()
            return frame
        except Exception as e:
            logging.error(f"detect_objects error: {e}", exc_info=True)
            return frame


# =============================================================================
#                      SETTINGS (YAML Functions Integrated)
# =============================================================================

def load_settings_from_yaml(parent=None):
    try:
        if not os.path.exists(config.settings_yml): return
        with open(config.settings_yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rtsp = data.get("rtsp", {}) or {}; api  = data.get("api", {}) or {}; lic  = data.get("license", {}) or {}
        for ch in CHANNELS:
            key = f"ch{ch}"
            config.rtsp_map[ch]      = rtsp.get(key, "")
            config.url_gates[ch]     = api.get(key, "")
            config.license_addrs[ch] = lic.get(key, "")
    except Exception as err:
        QMessageBox.critical(parent, "設定檔錯誤", f"讀取 settings.yml 發生錯誤：\n{err}")

def save_settings_to_yaml(rtsp_map, api_map, lic_map, parent=None):
    try:
        os.makedirs(os.path.dirname(config.settings_yml), exist_ok=True)
        data = {"rtsp": {}, "api": {}, "license": {}}
        for ch in CHANNELS:
            k = f"ch{ch}"
            data["rtsp"][k]    = rtsp_map.get(ch, "")
            data["api"][k]     = api_map.get(ch, "")
            data["license"][k] = lic_map.get(ch, "")
        with open(config.settings_yml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"寫入 settings.yml 失敗：\n{e}")
        raise # Re-raise to stop the dialog from closing successfully

# =============================================================================
#                           UI & MAIN APPLICATION
# =============================================================================
class PySideUI:
    def __init__(self): self.design_w, self.design_h = 1920, 1080; self.sx, self.sy = 1.0, 1.0
    def _get_work_area(self): return QApplication.primaryScreen().availableGeometry()
    def _set_scaling(self, w, h): self.sx = w / self.design_w; self.sy = h / self.design_h
    def _S(self, val, axis="x"): return int(round(val * (self.sx if axis == "x" else self.sy)))
    def _FS(self, fs): return max(10, int(round(fs * self.sy)))

class MainApplication(QMainWindow):
    status_update_signal = Signal(str, str, str)

    def __init__(self, executor):
        super().__init__()
        self.executor = executor
        self.ui = PySideUI()
        self.open_btn_by_ch, self.label_by_ch, self.tile_by_ch = {}, {}, {}
        
        ### --- NEW/MODIFIED --- ###
        # Connect the signal to the update slot
        self.status_update_signal.connect(self.update_status_display)

        self.init_ui()
        self.init_streams()
        self.update_timer = QTimer(self); self.update_timer.timeout.connect(self.update_ui_loop); self.update_timer.start(50)

    def init_ui(self):
        cols, rows = self.compute_grid_dims(NUM_CHANNELS)
        need_w, need_h = 1920, 1080
        rect = self.ui._get_work_area()
        final_w, final_h = min(need_w, rect.width()), min(need_h, rect.height())
        self.ui._set_scaling(final_w, final_h)
        self.setGeometry(rect.x(), rect.y(), final_w, final_h)
        self.setWindowTitle(f"ALPR {NUM_CHANNELS}路車牌辨識系統") # <-- 修改: 標題
        self.setWindowIcon(QIcon(config.icon_path))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 建立一個主水平佈局
        main_layout = QHBoxLayout(self.central_widget)

        # --- 左側面板 ---
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(self.ui._S(450)) # 設定左側面板固定寬度

        # 狀態群組框
        # --- 修改: 以下為繁體中文翻譯 ---
        status_groupbox = QGroupBox("最新車輛狀態")
        status_layout = QFormLayout(status_groupbox)
        self.status_plate_label = QLabel("N/A")
        self.status_state_label = QLabel("等待偵測中...")
        font = self.status_plate_label.font()
        font.setPointSize(self.ui._FS(14))
        self.status_plate_label.setFont(font)
        self.status_state_label.setFont(font)
        status_layout.addRow("車牌號碼:", self.status_plate_label)
        status_layout.addRow("狀態:", self.status_state_label)
        left_panel_layout.addWidget(status_groupbox)
        left_panel_layout.setStretchFactor(status_groupbox, 0) # 固定大小，不伸展

        # 日誌群組框
        self.log_groupbox = QGroupBox("偵測日誌") # <-- 修改: 標題
        self._build_log_table(self.log_groupbox)
        left_panel_layout.addWidget(self.log_groupbox)
        left_panel_layout.setStretchFactor(self.log_groupbox, 1) # 允許垂直伸展以填滿空間

        main_layout.addWidget(left_panel)

        # --- CCTV 影像網格 (右側面板) ---
        self.cctv_grid_widget = QWidget()
        main_layout.addWidget(self.cctv_grid_widget, 1) # 加入網格並使其水平伸展
        self.build_cctv_grid(self.cctv_grid_widget, cols, rows)
        
        self.init_log_table()

    def _build_log_table(self, parent):
        layout = QVBoxLayout(parent)
        self.log_tree = QTreeWidget(); self.log_tree.setHeaderLabels(["日期", "時間", "車牌", "通道"])
        self.log_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.log_tree)

    def init_log_table(self):
        try:
            self.log_tree.clear()
            for row in controlManager.query_recent_logs(limit=100): self.log_tree.addTopLevelItem(QTreeWidgetItem(list(row)))
        except Exception as e: logging.error(f"init_log_table error: {e}")

    def build_cctv_grid(self, parent, cols, rows):
        # This function now correctly uses a grid layout passed from the parent
        grid_layout = QGridLayout(parent)
        target_w, target_h = (480, 420) if NUM_CHANNELS <= 4 else (360, 360)
        btn_h, btn_margin = 30, 8

        for i, ch in enumerate(CHANNELS):
            if i >= cols * rows: break
            r, c = divmod(i, cols)
            
            # Create a tile widget with its own vertical layout
            tile = QWidget()
            tile_layout = QVBoxLayout()
            tile.setLayout(tile_layout)
            self.tile_by_ch[ch] = tile
            
            # Create the label for the video feed
            lbl = QLabel()
            lbl.setStyleSheet("background-color: black; color: white;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setText(f"CCTV {ch}")
            lbl.setMinimumSize(self.ui._S(target_w * 0.5), self.ui._S(target_h * 0.5))
            self.label_by_ch[ch] = lbl
            
            # Create the "Open Door" button
            btn = QPushButton("開門")
            btn.setFixedSize(self.ui._S(100), self.ui._S(btn_h))
            btn.clicked.connect(lambda checked, c=ch: self.on_open_channel(c))
            self.open_btn_by_ch[ch] = btn

            # Add widgets to the tile's internal layout
            tile_layout.addWidget(lbl, 1) # Label takes expanding space
            tile_layout.addWidget(btn, 0, Qt.AlignCenter) # Button is centered below
            
            # Add the complete tile to the main grid
            grid_layout.addWidget(tile, r, c)
    
    def _set_label_error_message(self, label: QLabel, message: str):
        """Draws an error message directly on a QLabel."""
        try:
            # It's possible the label is not yet sized when this is first called
            size = label.size()
            if not size.isValid() or size.width() < 10 or size.height() < 10:
                size = QSize(200, 150) # Use a sensible default if not yet visible

            pixmap = QPixmap(size)
            pixmap.fill(Qt.GlobalColor.red)
            painter = QPainter(pixmap)
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("SimSun", max(12, size.height() // 15)))
            painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.TextWordWrap, message)
            painter.end()
            label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"_set_label_error_message failed: {e}")

    def init_streams(self):
        for ch in CHANNELS:
            src = config.rtsp_map.get(ch, '')
            if not src: 
                # Now, it safely sets an error message on the UI from the main thread
                if lbl := self.label_by_ch.get(ch):
                    self._set_label_error_message(lbl, f"通道 {ch}\n未設定來源")
                continue
            
            config.video_streams.append(VideoCapture(
                rtsp_url=src, 
                label_widget=self.label_by_ch.get(ch), 
                main_window=self,
                ch=ch, 
                executor=self.executor
            ))

    def update_ui_loop(self):
        for vs in config.video_streams:
            frame = vs.read()
            if frame is not None:
                processed = vs.detect_objects(frame)
                vs.show_frame_on_label(processed)

    ### --- NEW --- ###
    @Slot(str, str, str)
    def update_status_display(self, plate, status, color):
        """This slot updates the status labels in the main thread."""
        self.status_plate_label.setText(plate)
        self.status_state_label.setText(status)
        self.status_state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    ### --- NEW --- ###
    def check_and_update_status(self, license_plate):
        """Worker function to query DB and emit signal with the result."""
        status_code, message = controlManager.query_vehicle_status(license_plate)
        
        color = "black"
        if status_code == "CAN_IN":
            color = "green"
        elif status_code == "ALREADY_PARKED":
            color = "orange"
        elif status_code == "NOT_REGISTERED":
            color = "red"
        
        self.status_update_signal.emit(license_plate, message, color)

    
    def on_open_channel(self, ch: str):
        url_gate = config.url_gates.get(ch)
        if not url_gate: return QMessageBox.critical(self, "Error", f"CH{ch} Gate API not set.")
        if btn := self.open_btn_by_ch.get(ch): btn.setEnabled(False)
        now = datetime.now()
        data = {"car_no":"MANUAL", "type":ch, "add_date":f"{now.year-1911}{now:%m%d}", "add_time":f"{now:%H%M%S}"}
        self.executor.submit(controlManager.door_open, data, url_gate, ch)
        QTimer.singleShot(1200, lambda: btn.setEnabled(True) if btn else None)

    def closeEvent(self, event):
        logging.info("Closing application..."); self.update_timer.stop()
        for vs in config.video_streams: vs.stop()
        self.executor.shutdown(wait=False); event.accept()
        
    def compute_grid_dims(self, n):
        if n <= 1: cols = 1
        elif n <= 4: cols = 2
        else: cols = min(GRID_MAX_COLS, 4)
        return cols, (n + cols - 1) // cols

def run_settings_dialog():
    """ 此函數已被重構以使用輔助函數 """
    dialog = QDialog()
    dialog.setWindowTitle("系統設定") # <-- 修改: 標題
    dialog.setMinimumSize(720, 800)
    
    # --- 使用輔助函數載入設定 ---
    load_settings_from_yaml(dialog)
    
    # --- UI 介面設定 ---
    layout = QVBoxLayout(dialog)
    scroll = QScrollArea()
    layout.addWidget(scroll)
    scroll.setWidgetResizable(True)
    content = QWidget()
    scroll.setWidget(content)
    form = QFormLayout(content)
    entries = {}

    # --- 修改: 以下為繁體中文翻譯 ---
    form.addRow(QLabel("<h3>RTSP 串流 / 相機索引</h3>"))
    for ch in CHANNELS:
        entries[f"rtsp_{ch}"] = QLineEdit(config.rtsp_map.get(ch, ""))
        form.addRow(QLabel(f"通道 {ch}:"), entries[f"rtsp_{ch}"])

    form.addRow(QLabel("<h3>閘門 API (http://...)</h3>"))
    for ch in CHANNELS:
        entries[f"api_{ch}"] = QLineEdit(config.url_gates.get(ch, ""))
        form.addRow(QLabel(f"通道 {ch}:"), entries[f"api_{ch}"])

    form.addRow(QLabel("<h3>字幕機 / LED 顯示器 (tcp://ip:port)</h3>"))
    for ch in CHANNELS:
        entries[f"lic_{ch}"] = QLineEdit(config.license_addrs.get(ch, ""))
        form.addRow(QLabel(f"通道 {ch}:"), entries[f"lic_{ch}"])
        
    submit_button = QPushButton("確認")
    submit_button.setFixedSize(120, 32) # <-- 設定固定大小 (寬度, 高度)
    layout.addWidget(submit_button, 0, Qt.AlignCenter)
    
    def submit():
        if not dialog.isVisible(): return
        
        # 從 UI 收集資料
        rtsp_map = {ch: entries[f"rtsp_{ch}"].text().strip() for ch in CHANNELS}
        api_map  = {ch: entries[f"api_{ch}"].text().strip() for ch in CHANNELS}
        lic_map  = {ch: entries[f"lic_{ch}"].text().strip() for ch in CHANNELS}
        
        # 更新全域設定
        config.rtsp_map, config.url_gates, config.license_addrs = rtsp_map, api_map, lic_map
        try:
            # 使用輔助函數儲存設定
            save_settings_to_yaml(rtsp_map, api_map, lic_map, parent=dialog)
            dialog.accept() # 成功後關閉對話框
        except Exception:
            # 儲存函數已顯示錯誤訊息
            pass 

    submit_button.clicked.connect(submit)
    QTimer.singleShot(10000, submit) # 10秒後自動提交
    return dialog.exec() == QDialog.Accepted
# =============================================================================
#                                 MAIN ENTRY POINT
# =============================================================================
def main():
    # 1. Create the application instance
    app = QApplication(sys.argv)

    # 2. Apply the modern material theme TO THE ENTIRE APP
    # This single line themes all windows and dialogs.
    apply_stylesheet(app, theme='dark_blue.xml')
    
    # 3. Create the thread pool
    executor = ThreadPoolExecutor(max_workers=max(4, NUM_CHANNELS * 2))
    
    # 4. Run the settings dialog (it will now be styled)
    if run_settings_dialog():
        # If settings are confirmed, create and show the main window (also styled)
        main_window = MainApplication(executor)
        config.root = main_window
        main_window.show()
        
        # 5. Start the application's event loop
        sys.exit(app.exec())
    else:
        # If settings are cancelled, exit gracefully
        logging.info("Settings not confirmed. Exiting.")
        executor.shutdown()

if __name__ == "__main__":
    main()