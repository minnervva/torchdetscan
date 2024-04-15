""" Example of a python module that has no non-deterministic functions """

# This is silly code intened to demonstrate how the linter deals with
# modules that have no non-deterministic functions.

import random

def generate_filler_code(num_lines=10):
    filler_code = []
    for _ in range(num_lines):
        indent_level = random.randint(0, 4)
        line = "    " * indent_level
        line += "print('Lorem ipsum dolor sit amet, consectetur adipiscing elit.')"
        filler_code.append(line)
    return "\n".join(filler_code)

print(generate_filler_code())