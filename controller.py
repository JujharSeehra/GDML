import time
import Quartz


class GDController:
    def __init__(self):
        self.held = False


    def _send_key(self, keycode):
        down = Quartz.CGEventCreateKeyboardEvent(None,keycode,True)
        up = Quartz.CGEventCreateKeyboardEvent(None,keycode,False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap,down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap,up)

    def toggle_pause(self):
        """
        Escape key on macOS.
        Geometry Dash uses this to pause/unpause.
        """
        self.release()
        self._send_key(53)
        time.sleep(0.05)
    def _mouse_position(self):
        event = Quartz.CGEventCreate(None)
        position = Quartz.CGEventGetLocation(event)
        return position.x, position.y

    def _send_mouse_event(self, event_type):
        x, y = self._mouse_position()
        event = Quartz.CGEventCreateMouseEvent(None, event_type, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def press(self):
        """Hold Geometry Dash input."""
        if not self.held:
            self._send_mouse_event(Quartz.kCGEventLeftMouseDown)
            self.held = True

    def release(self):
        """Release Geometry Dash input."""
        if self.held:
            self._send_mouse_event(Quartz.kCGEventLeftMouseUp)
            self.held = False

    def click(self, duration=0.05):
        """Perform a normal Geometry Dash click."""
        self.press()
        time.sleep(duration)
        self.release()

    def reset(self):
        self.release()


if __name__ == "__main__":
    controller = GDController()
    print("Switch to Geometry Dash!")
    time.sleep(3)
    print("HOLDING...")
    controller.press()
    time.sleep(6)
    print("RELEASING...")
    controller.release()
    print("Finished.")