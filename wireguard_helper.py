
import ctypes
import os
import subprocess
import sys
import time
 
import re
from logger import get_logger
 


LOG = get_logger("wire_guard")

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def sc(args):
    cp = subprocess.run(["sc"] + args, capture_output=True, text=True)
    return cp.returncode, cp.stdout + cp.stderr

def is_wireguard_running(tunnel_name:str):
    svc = f"WireGuardTunnel${tunnel_name}"
    rc, out = sc(["query", svc])
    m = re.search(r"STATE\s*:\s*\d+\s+(\w+)", out)
    state = m.group(1) if m else "UNKNOWN"
    if state == "RUNNING":
        return True
    else:
        return False
    

def switch_wire_guard(tunnel_name):
    if is_wireguard_running(tunnel_name):
        toggle_vpn(tunnel_name, "stop")
    else:
        toggle_vpn(tunnel_name, "start")

def toggle_vpn(tunnel_name, action="start"):
    # Path to the WireGuard executable
    #wg_path = r"C:\Program Files\WireGuard\wg.exe"
   
    try:
        # This command tells the background service to start the tunnel
        #subprocess.run([wg_path, action, tunnel_name], check=True)
        subprocess.run(["sc", action, f"WireGuardTunnel${tunnel_name}"], check=False)
        time.sleep(5)
        LOG.info(f"Tunnel {tunnel_name} is now {action}.")


    except subprocess.CalledProcessError as e:
        LOG.info(f"Failed: {e}. You might need to be in the 'Network Configuration Operators' group.")

def run_wireguard_command(args):
    wireguard_path = r'C:\Program Files\WireGuard\wireguard.exe'
    if not os.path.exists(wireguard_path):
        LOG.info("Error: wireguard.exe not found at", wireguard_path)
        LOG.info("Make sure WireGuard is installed.")
        sys.exit(1)

    cmd = [wireguard_path] + args
    try:
        if not is_admin():
            LOG.info("Relaunching as admin (required for tunnel operations)...")
            ctypes.windll.shell32.ShellExecuteW(None, "runas", cmd[0], " ".join(cmd[1:]), None, 1)
            LOG.info(" not admin id.")
            sys.exit(0)

        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            LOG.info("Success:", result.stdout.strip() or "(no output)")
            if result.stderr:
                LOG.info("Notes:", result.stderr.strip())
    except subprocess.CalledProcessError as e:
        LOG.info(f"Failed (exit code {e.returncode}):")
        LOG.info(e.stderr.strip() or e.stdout.strip() or "(no error details)")
    except Exception as e:
        LOG.info("Unexpected error:", str(e))

def get_config_path(tunnel_name: str, use_dpapi: bool = True) -> str:
    """Build the expected full path to the config file."""
    ext = ".conf.dpapi" if use_dpapi else ".conf"
    return rf'C:\Program Files\WireGuard\Data\Configurations\{tunnel_name}{ext}'

def activate_tunnel_1(tunnel_name: str):
    config_path = get_config_path(tunnel_name)
    if not os.path.exists(config_path):
        LOG.info(f"Config not found: {config_path}")
        LOG.info("Check the exact name in WireGuard GUI (case-sensitive).")
        LOG.info("Tip: Look in C:\\Program Files\\WireGuard\\Data\\Configurations\\")
        sys.exit(1)
    
    LOG.info(f"Activating tunnel '{tunnel_name}' → {config_path}")
    run_wireguard_command(['/installtunnelservice', config_path])

def deactivate_tunnel_1(tunnel_name: str):
    config_path = get_config_path(tunnel_name)
    LOG.info(f"Deactivating tunnel '{tunnel_name}'")
    run_wireguard_command(['/uninstalltunnelservice', tunnel_name])


def activate_tunnel(tunnel_name: str):
    toggle_vpn(tunnel_name, action="start")

def deactivate_tunnel(tunnel_name: str):
    toggle_vpn(tunnel_name, action="stop")

