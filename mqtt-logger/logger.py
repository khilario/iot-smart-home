"""
IoT Smart Home - MQTT to SQLite Logger

Listens for smart-home telemetry over MQTT and stores each message
in a local SQLite database for later querying and visualization.

Expected MQTT payload:

{
    "count": 42,
    "node": "room1",
    "sensor": "lights",
    "status": "on"
}
"""

import json
import sqlite3
from datetime import datetime

import paho.mqtt.client as mqtt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "smart-home/status"

DATABASE_FILE = "home_system.db"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def initialize_database():
    """Create the telemetry table if it does not already exist."""

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                real_time TEXT NOT NULL,
                counter INTEGER,
                node_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL
            )
            """
        )


database_connection = sqlite3.connect(
    DATABASE_FILE,
    check_same_thread=False,
)


def normalize_status(status):
    """
    Convert incoming status values into database-friendly values.

    Examples:
        "on"  -> 1
        "off" -> 0
        23    -> 23
        21.5  -> 21.5
    """

    if isinstance(status, str):
        status = status.strip().lower()

        if status == "on":
            return 1

        if status == "off":
            return 0

        if status == "na":
            return None

        try:
            return float(status)

        except ValueError:
            return None

    if isinstance(status, (int, float)):
        return status

    return None


def store_message(count, node, sensor, status):
    """Insert one telemetry message into SQLite."""

    value = normalize_status(status)

    try:
        database_connection.execute(
            """
            INSERT INTO logs (
                real_time,
                counter,
                node_id,
                sensor_type,
                value
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                count,
                node,
                sensor,
                value,
            ),
        )

        database_connection.commit()

        print(f"Stored: {node} / {sensor} = {value}")

    except sqlite3.Error as error:
        print(f"Database error: {error}")


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

def on_connect(client, userdata, flags, rc):
    """Subscribe to the telemetry topic after connecting."""

    if rc == 0:
        print("Connected to MQTT broker")

        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to: {MQTT_TOPIC}")

    else:
        print(f"MQTT connection failed with code {rc}")


def on_message(client, userdata, message):
    """Parse and store an incoming smart-home telemetry message."""

    try:
        payload = json.loads(message.payload.decode("utf-8"))

        count = payload["count"]
        node = payload["node"]
        sensor = payload["sensor"]
        status = payload["status"]

        store_message(
            count=count,
            node=node,
            sensor=sensor,
            status=status,
        )

    except json.JSONDecodeError:
        print("Received invalid JSON")

    except KeyError as error:
        print(f"Missing required field: {error}")

    except Exception as error:
        print(f"Unable to process MQTT message: {error}")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Start the MQTT telemetry logger."""

    initialize_database()

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print(
            f"Connecting to MQTT broker "
            f"{MQTT_BROKER}:{MQTT_PORT}..."
        )

        client.connect(
            MQTT_BROKER,
            MQTT_PORT,
            keepalive=60,
        )

        print("Smart Home logger running. Press Ctrl+C to stop.")

        client.loop_forever()

    except KeyboardInterrupt:
        print("\nStopping logger...")

    finally:
        client.disconnect()
        database_connection.close()

        print("Database connection closed.")


if __name__ == "__main__":
    main()