import socket
from typing import List


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def scan_local_ports(start: int = 7000, end: int = 9000) -> List[int]:
    open_ports: List[int] = []
    for port in range(start, end + 1):
        if is_port_open("127.0.0.1", port):
            open_ports.append(port)
    return open_ports

