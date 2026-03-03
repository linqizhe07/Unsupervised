import re


TMIN = 50
TMAX = 400
FITNESS_SLOPE = -1 / 700
FITNESS_INTERCEPT = 75 / 70


def _parse_episode_result(line):
    success_match = re.search(r"Success=(True|False)", line)
    if success_match is None:
        return None

    step_match = re.search(r"step\s+(\d+):", line)
    if step_match is None:
        step_match = re.search(r"Episode\s+(\d+):", line)
    if step_match is None:
        raise ValueError(f"Unable to parse steps from log line: {line.strip()}")

    steps = int(step_match.group(1))
    success = success_match.group(1) == "True"
    return steps, success


def calculate_fitness_score(log_file):
    total_fitness = 0.0
    num_episodes = 0

    with open(log_file, "r") as file:
        for line in file:
            parsed = _parse_episode_result(line)
            if parsed is None:
                continue

            steps, success = parsed
            if success:
                clipped_steps = min(max(steps, TMIN), TMAX)
                fitness = FITNESS_SLOPE * clipped_steps + FITNESS_INTERCEPT
                fitness = min(max(fitness, 0.5), 1.0)
            else:
                fitness = 0.0

            total_fitness += fitness
            num_episodes += 1

    if num_episodes > 0:
        average_fitness = total_fitness / num_episodes
    else:
        average_fitness = 0.0

    return average_fitness


# Example usage:
# average_fitness = calculate_fitness_score("performance_log.txt")
# print(f"Average Fitness Score: {average_fitness}")
