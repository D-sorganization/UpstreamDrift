with open("src/shared/python/pendulum_simulator/physics_base.py", "r") as f:
    content = f.read()

search = """
    # Cumulative mass from tip backwards: mass_below[i] = sum(masses[i:])
    # Each segment i contributes: -mass_below_i * g * L_i * cos(angle_i)
    # But for pendulums, the contribution is the total mass that passes
    # through segment i times the height change of that segment.
    V = 0.0
    for i in range(n):
        # ⚡ Bolt: arr.sum() is ~3x faster than np.sum(arr)
        mass_below = float(masses[i:].sum())
        V -= mass_below * g * lengths[i] * np.cos(absolute_angles[i])
"""

replace = """
    # Cumulative mass from tip backwards: mass_below[i] = sum(masses[i:])
    # Each segment i contributes: -mass_below_i * g * L_i * cos(angle_i)
    # But for pendulums, the contribution is the total mass that passes
    # through segment i times the height change of that segment.
    V = 0.0
    # ⚡ Bolt: Vectorized cumulative sum is O(n) instead of O(n^2) loop sum
    masses_arr = np.asarray(masses)
    mass_below = np.cumsum(masses_arr[::-1])[::-1]

    for i in range(n):
        V -= float(mass_below[i]) * g * lengths[i] * np.cos(absolute_angles[i])
"""
content = content.replace(search, replace)
with open("src/shared/python/pendulum_simulator/physics_base.py", "w") as f:
    f.write(content)
