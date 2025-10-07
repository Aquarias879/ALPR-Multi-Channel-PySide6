import struct


class PacketCreator:
    COLOR_MAP = {
        "red": bytes([0xFF, 0x00, 0x00]),
        "green": bytes([0x00, 0xFF, 0x00]),
        "white": bytes([0xFF, 0xFF, 0xFF]),
        "blue": bytes([0x00, 0x00, 0xFF]),
        "yellow": bytes([0xFF, 0xFF, 0x00]),
    }
    def __init__(self, license,color_name):
        """
        Initialize the PacketCreator with the provided license.

        :param license: License string to include in the packet.
        """
        # Parameters for the packet
        self.device_id_str = '00606edf969a'  # Device ID as a string
        self.card_id = 0x01                  # Specific device
        self.confirmation_flag = 0x01        # Request confirmation

        # Display settings
        self.window_number = 0x00            # Window Number (0 to 7)
        self.method = 0x00                   # Immediate display
        self.alignment = 0x01                # Left-top alignment
        self.speed = 0x03                    # Speed (1 to 100)
        self.dwell_time = 5                  # Dwell time in seconds
        self.font_size = 0x02                # Font size code
        self.font_type = 0x03                # Font type code
        self.font = (self.font_type << 4) | self.font_size  # Combine font size and type
        #self.color_rgb = bytes([0xFF, 0x00, 0x00])          # Red color (R, G, B)
        self.color_rgb = self.COLOR_MAP.get(color_name.lower(), self.COLOR_MAP["red"])

        # License and text
        self.text = license  # Traditional Chinese text

    def escape_encode(self, data):
        """
        Encode data by escaping special characters (0xA5, 0xAA, 0xAE) as per protocol.
        """
        escaped = bytearray()
        for byte in data:
            if byte == 0xA5:
                escaped.extend([0xAA, 0x05])  # 0xA5 → 0xAA 0x05
            elif byte == 0xAE:
                escaped.extend([0xAA, 0x0E])  # 0xAE → 0xAA 0x0E
            elif byte == 0xAA:
                escaped.extend([0xAA, 0x0A])  # 0xAA → 0xAA 0x0A
            else:
                escaped.append(byte)
        return escaped

    def create_packet(self):
        """
        Create the packet based on the initialized parameters.

        :return: Bytearray representing the packet.
        """
        # Start building the unescaped packet
        unescaped_packet = bytearray()
        unescaped_packet.append(0x68)  # Packet Type
        unescaped_packet.append(0x32)  # Card Type Code

        # Card ID
        unescaped_packet.append(self.card_id)

        # Protocol Code for CC=0x12 command
        protocol_code = 0x7B
        unescaped_packet.append(protocol_code)

        # Additional Info/Confirmation Flag
        unescaped_packet.append(self.confirmation_flag)

        # Build the CC data
        cc_data = bytearray()
        cc_data.append(0x12)               # CC code
        cc_data.append(self.window_number) # Window Number
        cc_data.append(self.method)        # Method (Display effect)
        cc_data.append(self.alignment)     # Alignment settings
        cc_data.append(self.speed)         # Speed
        # Dwell time (2 bytes, big endian)
        cc_data.extend(struct.pack('>H', self.dwell_time))
        cc_data.append(self.font)          # Font settings
        cc_data.extend(self.color_rgb)     # Text color (R, G, B)

        # Encode the text using Big5 with a null terminator
        text_bytes = self.text.encode('big5') + b'\x00'

        # Append escaped CC data
        cc_data.extend(text_bytes)

        # Calculate Packet Data Length LL LH (length of CC data + PO + TP), little endian
        packet_data_length = len(cc_data) + 2  # Include PO and TP
        unescaped_packet.extend(packet_data_length.to_bytes(2, byteorder='little'))

        # Packet Sequence Numbers (PO and TP)
        unescaped_packet.extend([0x00, 0x00])  # PO and TP

        # Append CC Data
        unescaped_packet.extend(cc_data)

        # Calculate checksum
        checksum = sum(unescaped_packet) & 0xFFFF  # 2-byte checksum
        checksum_bytes = checksum.to_bytes(2, byteorder='little')

        # Escape checksum
        escaped_checksum = self.escape_encode(checksum_bytes)

        # Build the final packet
        packet = bytearray()
        packet.append(0xA5)  # Start Code
        # Append Device ID (ASCII encoded, terminated with 0x00)
        device_id_bytes = self.device_id_str.encode('ascii') + b'\x00'
        packet.extend(self.escape_encode(device_id_bytes))
        packet.extend(self.escape_encode(unescaped_packet))  # Escaped unescaped_packet
        packet.extend(escaped_checksum)  # Escaped checksum
        packet.append(0xAE)  # End Code

        return packet

    def get_packet_hex(self):
        """
        Get the hexadecimal string representation of the packet.

        :return: Hexadecimal string of the packet.
        """
        packet = self.create_packet()
        return ' '.join('{:02X}'.format(b) for b in packet)
