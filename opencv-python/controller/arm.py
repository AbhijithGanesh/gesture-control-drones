from pymavlink import mavutil
import time


class MavHandler:
    def __init__(self):
        self.mav = mavutil.mavlink_connection("udp:127.0.0.1:14550")
        return None

    def handle_arming(self) -> bool:
        print("Testing for heartbeat")
        self.mav.wait_heartbeat()
        print("Waiting for the vehicle to arm")
        print("Sleeping for 20 seconds to allow the IMUs and the GPSs to load")
        time.sleep(20)
        self.mav.arducopter_arm()
        print("Waiting for arm ....")
        self.mav.motors_armed_wait()
        print("Armed!")
        return True

    def handle_mode(self, mode) -> bool:
        print("Checking for heartbeat")
        self.mav.wait_heartbeat()
        self.mav.set_mode(mode)
        print(f"Setting mode {mode}")
        return True

    def handle_takeoff(self, target_location) -> bool:
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            confirmation=0,
            param1=0,  # Param1 (not used for takeoff, set to 0)
            param2=0,  # Param2 (not used for takeoff, set to 0)
            param3=0,  # Param3 (not used for takeoff, set to 0)
            param4=0,  # Param4 (not used for takeoff, set to 0)
            param5=target_location[0],  # Latitude
            param6=target_location[1],  # Longitude
            param7=target_location[2],  # Altitude
        )
        time.sleep(10)
        return True
