#!/usr/bin/env python3
"""
Read initial_state from a TChem-atm YAML file and write the species as flat
key-value pairs into the initial_conditions block of a companion input.yaml.

The initial_state section already contains M, N2, and H2O, so they are
included automatically.

Usage:
    python3 gen_initial_conditions.py <path-to-uci_v2.yaml> [<path-to-input.yaml>]

If <path-to-input.yaml> is omitted, the script looks for input.yaml in the
same directory as this script.

The script replaces everything between:
    topography_filename: ...
and:
    ...   (the final YAML document-end marker)

with the species lines.
"""

import sys
import os
import yaml


# Markers used to locate the region to replace in input.yaml
_START_MARKER = "topography_filename:"
_END_MARKER   = "..."


def _make_loader():
    """Return a SafeLoader subclass that does not coerce bare words like
    'NO', 'YES', 'TRUE', 'FALSE' into Python booleans."""
    loader = yaml.SafeLoader
    # Build a patched class so we don't mutate the global SafeLoader
    class _Loader(loader):
        pass
    # Remove the boolean implicit resolver so species names like 'NO' stay strings
    _Loader.yaml_implicit_resolvers = {
        key: [(tag, regexp) for tag, regexp in resolvers
              if tag != "tag:yaml.org,2002:bool"]
        for key, resolvers in loader.yaml_implicit_resolvers.items()
    }
    return _Loader


def _load_species(chem_file):
    """Return an ordered list of (name, value) from initial_state in the given
    TChem-atm chemistry YAML file."""
    with open(chem_file, "r") as fh:
        data = yaml.load(fh, Loader=_make_loader())

    entries = []
    for name, info in data.get("initial_state", {}).items():
        value = info["initial_value"][0]
        entries.append((name, value))

    return entries


def _format_value(value):
    """Format a numeric value as a float, preserving scientific notation where appropriate."""
    # Cast integers to float so values like 1000 are written as 1000.0
    return repr(float(value))


def _build_species_block(entries):
    """Build the YAML lines for each species."""
    lines = []
    for name, value in entries:
        lines.append(f"  {name}: {_format_value(value)}")
    return lines


def _update_input_yaml(input_yaml_path, species_lines):
    """Replace the TODO block and old species entries in input.yaml in-place."""
    with open(input_yaml_path, "r") as fh:
        original = fh.readlines()

    # Find the line containing _START_MARKER and the final _END_MARKER
    start_idx = None
    end_idx   = None
    for i, line in enumerate(original):
        stripped = line.strip()
        if stripped.startswith(_START_MARKER) and start_idx is None:
            start_idx = i
        if stripped == _END_MARKER:
            end_idx = i  # keep updating so we get the last occurrence

    if start_idx is None or end_idx is None:
        raise RuntimeError(
            f"Could not locate '{_START_MARKER}' or '{_END_MARKER}' in {input_yaml_path}"
        )

    # Keep everything up to and including the topography_filename line,
    # then insert the species block, then the end marker.
    new_lines = (
        original[: start_idx + 1]
        + [line + "\n" for line in species_lines]
        + [original[end_idx]]
    )

    with open(input_yaml_path, "w") as fh:
        fh.writelines(new_lines)

    print(f"Updated {input_yaml_path} with {len(species_lines)} species entries.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    chem_file = sys.argv[1]

    if len(sys.argv) >= 3:
        input_yaml = sys.argv[2]
    else:
        input_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.yaml")

    entries = _load_species(chem_file)
    species_lines = _build_species_block(entries)
    _update_input_yaml(input_yaml, species_lines)


if __name__ == "__main__":
    main()
