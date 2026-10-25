#!/usr/bin/env python3
"""
ROS2 node that translates joystick inputs into HTTP commands for the rover.
Verified axis mapping:
  axes[0]    Left stick X  → turn    (0.0 rest, ±1.0 full)
  axes[2]    LT            → reverse (1.0 rest, -1.0 full)
  axes[5]    RT            → forward (1.0 rest, -1.0 full)
  buttons[4] LB            → enable input (must hold to drive)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import socket
import urllib.parse
import json
import threading

ROVER_IP   = '10.224.13.53'  # Rover IP
ROVER_PORT = 80

class PreconnectPool:

    def __init__(self, host: str, port: int):
        self.host  = host
        self.port  = port
        self._sock = None
        self._lock = threading.Lock()
        self._reconnect()          # pre-connect immediately on startup

    def _reconnect(self):
        threading.Thread(target=self._do_connect, daemon=True).start()

    def _do_connect(self):
        try:
            s = socket.create_connection((self.host, self.port), timeout=3.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no Nagle buffering
            with self._lock:
                if self._sock:
                    try: self._sock.close()
                    except Exception: pass
                self._sock = s
        except Exception:
            pass  # will retry on next send

    def send(self, data: bytes) -> bool:
        with self._lock:
            s, self._sock = self._sock, None   # grab and clear atomically

        if s is None:
            # No pre-connected socket ready — connect inline (fallback)
            self._reconnect()
            return False

        try:
            s.sendall(data)
        except Exception:
            pass
        finally:
            try: s.close()
            except Exception: pass

        self._reconnect()          # immediately pre-connect for next command
        return True

    def close(self):
        with self._lock:
            if self._sock:
                try: self._sock.close()
                except Exception: pass
                self._sock = None


def build_http_request(ip: str, json_str: str) -> bytes:
    encoded = urllib.parse.quote(json_str, safe='')
    req = (
        f"GET /js?json={encoded} HTTP/1.0\r\n"
        f"Host: {ip}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return req.encode()


class JoyRoverHttp(Node):
    def __init__(self):
        super().__init__('joy_rover_http')

        self.declare_parameter('rover_ip',       ROVER_IP)
        self.declare_parameter('axis_rt',        5)
        self.declare_parameter('axis_lt',        2)
        self.declare_parameter('axis_angular',   0)
        self.declare_parameter('max_speed',      0.45)
        self.declare_parameter('angular_scale',  0.35)
        self.declare_parameter('enable_button',  4)
        self.declare_parameter('require_enable', True)
        self.declare_parameter('cmd_hz',         20.0)  # 20 Hz — safe now

        self._L = 0.0
        self._R = 0.0
        self._lock = threading.Lock()

        ip = self.get_parameter('rover_ip').value
        self._pool = PreconnectPool(ip, ROVER_PORT)

        self.sub   = self.create_subscription(Joy, 'joy', self.joy_cb, 10)
        hz         = self.get_parameter('cmd_hz').value
        self.timer = self.create_timer(1.0 / hz, self.send_cmd)

        self.get_logger().info(
            f'Ready → http://{ip}/  |  pre-connecting...  |  hold LB to drive'
        )

    def joy_cb(self, msg: Joy):
        p = {
            'rt':      self.get_parameter('axis_rt').value,
            'lt':      self.get_parameter('axis_lt').value,
            'ang':     self.get_parameter('axis_angular').value,
            'btn':     self.get_parameter('enable_button').value,
            'req':     self.get_parameter('require_enable').value,
            'max_spd': self.get_parameter('max_speed').value,
            'ang_scl': self.get_parameter('angular_scale').value,
        }

        if p['req'] and not msg.buttons[p['btn']]:
            with self._lock:
                self._L = 0.0
                self._R = 0.0
            return

        rt       = (1.0 - msg.axes[p['rt']]) / 2.0
        lt       = (1.0 - msg.axes[p['lt']]) / 2.0
        throttle = (rt - lt) * p['max_spd']
        turn     =  msg.axes[p['ang']] * p['ang_scl']

        clamp = lambda v: max(-p['max_spd'], min(p['max_spd'], v))

        with self._lock:
            self._L = round(clamp(throttle - turn), 3)
            self._R = round(clamp(throttle + turn), 3)

    def send_cmd(self):
        with self._lock:
            L, R = self._L, self._R

        ip      = self.get_parameter('rover_ip').value
        payload = json.dumps({"T": 1, "L": L, "R": R})
        data    = build_http_request(ip, payload)

        sent = self._pool.send(data)
        if not sent:
            self.get_logger().warn(
                'Socket not ready (still connecting)', throttle_duration_sec=2.0
            )

    def stop_rover(self):
        ip      = self.get_parameter('rover_ip').value
        payload = json.dumps({"T": 1, "L": 0, "R": 0})
        data    = build_http_request(ip, payload)
        self._pool.send(data)
        self._pool.close()
        self.get_logger().info('Stop command sent.')


def main(args=None):
    rclpy.init(args=args)
    node = JoyRoverHttp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_rover()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()