#!/usr/bin/env python3

# Quick script to fix the raw \n characters in the visualization function

import re

# Read the file
with open('/home/andrew99245/SAKURA_Reasoning/src/models/hal_inference_angle_correlation.py', 'r') as f:
    content = f.read()

# Fix the raw \n characters in string literals
content = re.sub(r'\\n\s+', '\n    ', content)

# Write back
with open('/home/andrew99245/SAKURA_Reasoning/src/models/hal_inference_angle_correlation.py', 'w') as f:
    f.write(content)

print("Fixed raw \\n characters in the file")