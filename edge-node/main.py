"""
IoT Smart Home - Raspberry Pi Pico W Edge Node

Controls local smart-home sensors and actuators while communicating
with a Raspberry Pi gateway over MQTT.

Hardware:
- Room 1 LED: GP16
- Room 2 LED: GP17
- Living room LED: GP13
- Garage alarm LED: GP22
- Room 1 button: GP28
- Room 2 button: GP27
- Alarm reset button: GP26
- IR sensor: GP14
- PIR sensor: GP18
- DHT11: GP15
- Buzzer: GP20
"""

from machine import Pin, PWM
import time
import dht
import network
import json

from simple import MQTTClient
from config import (
    WIFI_SSID,
    WIFI_PASSWORD,
    MQTT_BROKER,
    MQTT_CLIENT_ID,
    MQTT_STATUS_TOPIC,
    MQTT_CONTROL_TOPIC,
)


# ---------------------------------------------------------------------------
# Hardware Configuration
# ---------------------------------------------------------------------------

room1_led = Pin(16, Pin.OUT)
room2_led = Pin(17, Pin.OUT)
living_room_led = Pin(13, Pin.OUT)
garage_alarm_led = Pin(22, Pin.OUT)

room1_button = Pin(28, Pin.IN, Pin.PULL_UP)
room2_button = Pin(27, Pin.IN, Pin.PULL_UP)
alarm_reset_button = Pin(26, Pin.IN, Pin.PULL_UP)

ir_sensor = Pin(14, Pin.IN)
pir_sensor = Pin(18, Pin.IN)

dht_sensor = dht.DHT11(Pin(15, Pin.IN, Pin.PULL_UP))

buzzer = PWM(Pin(20))
buzzer.duty_u16(0)


# ---------------------------------------------------------------------------
# Timing Configuration
# ---------------------------------------------------------------------------

DEBOUNCE_MS = 250
DHT_SAMPLE_PERIOD_MS = 2000
PIR_WARMUP_MS = 5000
ALARM_BLINK_PERIOD_MS = 250
LIVING_ROOM_TIMEOUT_MS = 10_000


# ---------------------------------------------------------------------------
# System State
# ---------------------------------------------------------------------------

last_button_press = {
    "room1": 0,
    "room2": 0,
    "alarm_reset": 0,
}

room1_on = False
room2_on = False

living_room_on = False
living_room_timeout = 0
living_room_remote_hold = False

last_temperature = None
last_humidity = None
last_dht_sample = 0

pir_ready = False
pir_warmup_start = time.ticks_ms()

motion_streak = 0
alarm_active = False
alarm_latched = False
last_alarm_blink = 0

system_start = time.ticks_ms()
last_status_publish = 0

mqtt_client = None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def button_debounced(button_name):
    """Return True when a button press passes the debounce interval."""
    now = time.ticks_ms()

    if time.ticks_diff(now, last_button_press[button_name]) > DEBOUNCE_MS:
        last_button_press[button_name] = now
        return True

    return False


def set_buzzer(enabled):
    """Enable or disable the garage alarm buzzer."""
    if enabled:
        buzzer.freq(1500)
        buzzer.duty_u16(30000)
    else:
        buzzer.duty_u16(0)


def publish_status(elapsed_seconds, node, sensor, status):
    """Publish a smart-home status message as JSON over MQTT."""
    payload = {
        "count": elapsed_seconds,
        "node": node,
        "sensor": sensor,
        "status": status,
    }

    message = json.dumps(payload)
    print(message)

    mqtt_client.publish(MQTT_STATUS_TOPIC, message)


# ---------------------------------------------------------------------------
# Physical Button Handlers
# ---------------------------------------------------------------------------

def room1_button_handler(pin):
    """Toggle the Room 1 light."""
    global room1_on

    if button_debounced("room1"):
        room1_on = not room1_on
        room1_led.value(room1_on)


def room2_button_handler(pin):
    """Toggle the Room 2 light."""
    global room2_on

    if button_debounced("room2"):
        room2_on = not room2_on
        room2_led.value(room2_on)


def alarm_reset_handler(pin):
    """Clear the latched garage alarm."""
    global alarm_active, alarm_latched, motion_streak

    if button_debounced("alarm_reset"):
        alarm_active = False
        alarm_latched = False
        motion_streak = 0

        set_buzzer(False)
        garage_alarm_led.value(0)


room1_button.irq(
    trigger=Pin.IRQ_FALLING,
    handler=room1_button_handler,
)

room2_button.irq(
    trigger=Pin.IRQ_FALLING,
    handler=room2_button_handler,
)

alarm_reset_button.irq(
    trigger=Pin.IRQ_FALLING,
    handler=alarm_reset_handler,
)


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

def connect_wifi():
    """Connect the Pico W to the configured Wi-Fi network."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    print("Connecting to Wi-Fi...")

    while not wlan.isconnected():
        time.sleep(1)

    print("Wi-Fi connected:", wlan.ifconfig()[0])

    return wlan


def connect_mqtt():
    """Connect to the MQTT broker."""
    global mqtt_client

    mqtt_client = MQTTClient(
        MQTT_CLIENT_ID,
        MQTT_BROKER,
    )

    mqtt_client.set_callback(handle_remote_command)
    mqtt_client.connect()
    mqtt_client.subscribe(MQTT_CONTROL_TOPIC)

    print("Connected to MQTT broker:", MQTT_BROKER)
    print("Listening on:", MQTT_CONTROL_TOPIC)


# ---------------------------------------------------------------------------
# Remote Control
# ---------------------------------------------------------------------------

def handle_remote_command(topic, message):
    """
    Process remote MQTT commands.

    Expected commands:
        room1:on
        room1:off
        room2:on
        room2:off
        livingroom:on
        livingroom:off
        garage:on
        garage:off
    """

    global room1_on
    global room2_on
    global living_room_on
    global living_room_remote_hold
    global alarm_active
    global alarm_latched
    global motion_streak

    try:
        command = message.decode().strip().lower()

        if ":" not in command:
            print("Invalid command:", command)
            return

        device, action = command.split(":", 1)

        if action not in ("on", "off"):
            print("Invalid action:", action)
            return

        enabled = action == "on"

        if device == "room1":
            room1_on = enabled
            room1_led.value(room1_on)

        elif device == "room2":
            room2_on = enabled
            room2_led.value(room2_on)

        elif device == "livingroom":
            living_room_remote_hold = enabled
            living_room_on = enabled
            living_room_led.value(living_room_on)

        elif device == "garage":
            if enabled:
                alarm_latched = True
                alarm_active = True
            else:
                alarm_latched = False
                alarm_active = False
                motion_streak = 0

                set_buzzer(False)
                garage_alarm_led.value(0)

        else:
            print("Unknown device:", device)
            return

        print("Remote command executed:", command)

    except Exception as error:
        print("MQTT command error:", error)


# ---------------------------------------------------------------------------
# Sensors and Automation
# ---------------------------------------------------------------------------

def update_pir_warmup(now):
    """Enable PIR processing after its startup stabilization period."""
    global pir_ready

    if not pir_ready:
        if time.ticks_diff(now, pir_warmup_start) >= PIR_WARMUP_MS:
            pir_ready = True
            print("PIR sensor ready")


def update_living_room(now):
    """Handle IR-triggered living room lighting."""
    global living_room_on
    global living_room_timeout
    global living_room_remote_hold

    ir_detected = ir_sensor.value() == 0

    if ir_detected:
        living_room_on = True
        living_room_remote_hold = False

        living_room_timeout = time.ticks_add(
            now,
            LIVING_ROOM_TIMEOUT_MS,
        )

        living_room_led.value(1)

    if living_room_on and not living_room_remote_hold:
        if time.ticks_diff(now, living_room_timeout) >= 0:
            living_room_on = False
            living_room_led.value(0)


def update_environment_sensor(now):
    """Sample temperature and humidity from the DHT11."""
    global last_temperature
    global last_humidity
    global last_dht_sample

    if time.ticks_diff(now, last_dht_sample) < DHT_SAMPLE_PERIOD_MS:
        return

    last_dht_sample = now

    try:
        dht_sensor.measure()

        last_temperature = dht_sensor.temperature()
        last_humidity = dht_sensor.humidity()

    except OSError:
        # Retain the previous valid reading.
        pass


def update_garage_alarm():
    """Update PIR motion detection and alarm state."""
    global motion_streak
    global alarm_active
    global alarm_latched

    if pir_ready and not alarm_latched:
        if pir_sensor.value() == 1:
            motion_streak += 1
        else:
            motion_streak = 0

        # Five consecutive detections at the 1 Hz update rate.
        if motion_streak >= 5:
            alarm_latched = True
            alarm_active = True

    if alarm_latched:
        alarm_active = True


def update_alarm_outputs(now):
    """Drive the buzzer and blinking garage alarm LED."""
    global last_alarm_blink

    if alarm_active:
        set_buzzer(True)

        if time.ticks_diff(now, last_alarm_blink) >= ALARM_BLINK_PERIOD_MS:
            last_alarm_blink = now
            garage_alarm_led.value(
                0 if garage_alarm_led.value() else 1
            )

    else:
        set_buzzer(False)
        garage_alarm_led.value(0)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def publish_system_status(now):
    """Publish the current system state once per second."""
    global last_status_publish

    if time.ticks_diff(now, last_status_publish) < 1000:
        return

    last_status_publish = now

    elapsed_seconds = time.ticks_diff(
        now,
        system_start,
    ) // 1000

    update_garage_alarm()

    publish_status(
        elapsed_seconds,
        "room1",
        "lights",
        "on" if room1_led.value() else "off",
    )

    publish_status(
        elapsed_seconds,
        "room2",
        "lights",
        "on" if room2_led.value() else "off",
    )

    publish_status(
        elapsed_seconds,
        "livingroom",
        "lights",
        "on" if living_room_led.value() else "off",
    )

    publish_status(
        elapsed_seconds,
        "thermostat",
        "temperature",
        "NA" if last_temperature is None else last_temperature,
    )

    publish_status(
        elapsed_seconds,
        "garage",
        "alarm",
        "on" if alarm_active else "off",
    )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Start networking and run the smart-home control loop."""

    connect_wifi()
    connect_mqtt()

    print("Smart Home edge node running")

    while True:
        mqtt_client.check_msg()

        now = time.ticks_ms()

        update_pir_warmup(now)
        update_living_room(now)
        update_environment_sensor(now)
        publish_system_status(now)
        update_alarm_outputs(now)

        time.sleep_ms(10)


if __name__ == "__main__":
    main()