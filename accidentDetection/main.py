import os
import serial
from twilio.rest import Client

# Twilio setup
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)
whatsapp_from = 'whatsapp:+14155238886'  # Twilio Sandbox number
whatsapp_to = 'whatsapp:+919310478147'

# Serial setup
arduino_serial = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust as needed

while True:
    if arduino_serial.in_waiting > 0:
        line = arduino_serial.readline().decode('utf-8').strip()
        if line == "Force detected":
            # Send WhatsApp message
            message = client.messages.create(
                body="Unusual force detected!",
                from_=whatsapp_from,
                to=whatsapp_to
            )
            print(f"Message sent: {message.sid}")
