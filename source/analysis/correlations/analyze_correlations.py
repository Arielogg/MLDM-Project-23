import re

pde_dict = {}

# Define variables and coefficients using regular expression
pattern = re.compile(r'([-+]?\s*\d*\.\d+\s*)?(\w+)(?:_(\d+))?')

with open(r"PDE.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.__contains__("20:"):
            key = '20'
        elif line.__contains__("22:"):
            key = '22'
        elif line.__contains__("24:"):
            key = '24'
        else:
            if line.strip():
                matches = pattern.findall(line)
                equation_dict = {}
                for coefficient, variable, exponent in matches:
                    coefficient = float(coefficient.replace(" ", "")) if coefficient else 1.0
                    exponent = int(exponent) if exponent else 1
                    equation_dict[variable + ('_' + exponent if exponent > 1 else '')] = [coefficient]
                if key not in pde_dict:
                    pde_dict[key] = equation_dict
                else:
                    dict1 = pde_dict[key]
                    for key_dict2, value_dict2 in equation_dict.items():
                        if key_dict2 in dict1:
                            dict1[key_dict2].extend(value_dict2)
                        else:
                            dict1[key_dict2] = value_dict2
                    pde_dict[key].update(dict1)

print(pde_dict)
