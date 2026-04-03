#!/usr/bin/env python3
"""
Compute molecular weights for Solution species in chem_mech.in using the
same atomic weights as chem_proc (mozpp.mods.f e_table / com_mass function).

Fixed species (M, N2, O2, H2O, H2, CH4, prsd_*) are excluded.

Usage: python compute_mw.py <chem_mech.in>

Output is a YAML block mapping suitable for pasting into input.yaml under
tchem_atm: molecular_weights:
"""

import re
import sys

# Atomic weights from chem_proc mozpp.mods.f e_table (iniele subroutine)
ATOMIC_WEIGHTS = {
    'H':  1.0074,
    'He': 4.0020602,
    'Li': 6.941,
    'Be': 9.012182,
    'B':  10.811,
    'C':  12.011,
    'N':  14.00674,
    'O':  15.9994,
    'F':  18.9984032,
    'Ne': 20.1797,
    'Na': 22.989768,
    'Mg': 24.305,
    'Al': 26.981539,
    'Si': 28.0855,
    'P':  30.97362,
    'S':  32.066,
    'Cl': 35.4527,
    'Ar': 39.948,
    'K':  39.0983,
    'Ca': 40.078,
    'Sc': 44.95591,
    'Ti': 47.867,
    'V':  50.9415,
    'Cr': 51.9961,
    'Mn': 54.93085,
    'Fe': 55.845,
    'Co': 58.9332,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.39,
    'Ga': 69.723,
    'Ge': 72.61,
    'As': 74.92159,
    'Se': 78.96,
    'Br': 79.904,
    'I':  126.90447,
}


def compute_mw(formula):
    """Compute molecular weight from a chemical formula string.

    Equivalent to chem_proc's com_mass() function. Parses standard chemical
    formula notation (e.g. CH4O2, AlSiO5, NaCl, C8520H11360O8520).
    """
    total = 0.0
    for match in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(
                f"Unknown element '{element}' in formula '{formula}'"
            )
        total += ATOMIC_WEIGHTS[element] * count
    return total


def parse_chem_mech(filepath):
    """Parse chem_mech.in and return solution species dict and fixed species set.

    Returns:
        solution_species: OrderedDict of {name: formula}
            alias formula used when species has '-> formula', else the name itself
        fixed_species: set of species names listed under 'Fixed'
    """
    solution_species = {}  # preserves insertion order (Python 3.7+)
    fixed_species = set()

    in_solution = False
    in_fixed = False

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if not stripped or stripped.startswith('*'):
                continue

            upper = stripped.upper()

            if upper == 'SOLUTION':
                in_solution = True
                in_fixed = False
                continue
            elif upper == 'END SOLUTION':
                in_solution = False
                continue
            elif upper == 'FIXED':
                in_fixed = True
                in_solution = False
                continue
            elif upper == 'END FIXED':
                in_fixed = False
                continue

            if in_fixed:
                for sp in stripped.split(','):
                    sp = sp.strip()
                    if sp:
                        fixed_species.add(sp)

            if in_solution:
                if '->' in stripped:
                    name, formula = stripped.split('->', 1)
                    name = name.strip()
                    formula = formula.strip()
                else:
                    name = stripped
                    formula = stripped  # symbol is used directly as formula
                solution_species[name] = formula

    return solution_species, fixed_species


def build_mw_lines(solution_species, fixed_species):
    lines = []
    for name, formula in solution_species.items():
        if name in fixed_species:
            continue
        mw = compute_mw(formula)
        lines.append(f"      {name}: {mw:.6f}")
    return lines


def update_yaml(yaml_file, mw_lines):
    """Replace the 'molecular_weights: [...]' line in yaml_file with the
    computed mapping block."""
    with open(yaml_file) as f:
        content = f.read()

    # Build new block: key line + indented entries
    new_block = "    molecular_weights:\n" + "\n".join(mw_lines)

    # Replace the molecular_weights line (handles both [] and existing mapping)
    # Match the key line and any subsequent indented lines belonging to it
    updated = re.sub(
        r'[ \t]*molecular_weights:[ \t]*\[?\]?\n(?:[ \t]+[^\n]*\n)*',
        new_block + "\n",
        content,
    )

    if updated == content:
        print("WARNING: 'molecular_weights' key not found in yaml file.",
              file=sys.stderr)
        return

    with open(yaml_file, 'w') as f:
        f.write(updated)
    print(f"Updated {yaml_file}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <chem_mech.in> [input.yaml]",
              file=sys.stderr)
        sys.exit(1)

    chem_mech_file = sys.argv[1]
    solution_species, fixed_species = parse_chem_mech(chem_mech_file)
    mw_lines = build_mw_lines(solution_species, fixed_species)

    if len(sys.argv) >= 3:
        update_yaml(sys.argv[2], mw_lines)
    else:
        print("    molecular_weights:")
        for line in mw_lines:
            print(line)


if __name__ == '__main__':
    main()
