#!/usr/bin/env python3
"""Portable replacement for: xxd -i -n <name> <input_file> <output_file>"""
import sys

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <name> <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    name, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(input_path, 'rb') as f:
        data = f.read()
    with open(output_path, 'w', newline='\n') as out:
        out.write(f'unsigned char {name}[] = {{\n')
        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_bytes = ', '.join(f'0x{b:02x}' for b in chunk)
            out.write(f'  {hex_bytes}{"," if i + 12 < len(data) else ""}\n')
        out.write(f'}};\n')
        out.write(f'unsigned int {name}_len = {len(data)};\n')

if __name__ == '__main__':
    main()
