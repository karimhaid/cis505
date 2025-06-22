import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Set


class Location:
    def __init__(self, x: float, y: float, cost: float = 1.0, weight: float = 1.0):
        self.x = x
        self.y = y
        self.cost = cost
        self.weight = weight

    def distance_to(self, other: "Location") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Location({self.x}, {self.y})"


class ChargingStationPlacer:
    def __init__(
        self,
        candidates: List[Location],
        demands: List[Location],
        service_radius: float = 2.0,
    ):
        self.candidates = candidates
        self.demands = demands
        self.service_radius = service_radius
        self.selected_stations = []
        self.covered_demands = set()

    def calculate_new_coverage(self, station: Location) -> float:
        """Calculate additional weighted coverage provided by a new station."""
        new_coverage = 0.0
        for i, demand in enumerate(self.demands):
            if (
                i not in self.covered_demands
                and station.distance_to(demand) <= self.service_radius
            ):
                new_coverage += demand.weight
        return new_coverage

    def update_coverage(self, station: Location):
        """Update the set of covered demands after placing a station."""
        for i, demand in enumerate(self.demands):
            if station.distance_to(demand) <= self.service_radius:
                self.covered_demands.add(i)

    def greedy_placement(self, k: int, verbose: bool = True) -> List[Location]:
        """
        Greedy algorithm for charging station placement.

        Args:
            k: Maximum number of stations to place
            verbose: Whether to print iteration details

        Returns:
            List of selected station locations
        """
        if verbose:
            print("=== EV Charging Station Placement Optimization ===\n")
            print(f"Input Parameters:")
            print(f"- Candidate locations: {len(self.candidates)}")
            print(f"- Demand points: {len(self.demands)}")
            print(f"- Maximum stations: {k}")
            print(f"- Service radius: {self.service_radius}\n")

        start_time = time.time()
        self.selected_stations = []
        self.covered_demands = set()

        for iteration in range(k):
            best_station = None
            best_score = float("-inf")

            if verbose:
                print(f"Iteration {iteration + 1}:")

            for candidate_idx, candidate in enumerate(
                self.candidates
            ):
                if candidate not in self.selected_stations:
                    new_coverage = self.calculate_new_coverage(candidate)
                    score = new_coverage - candidate.cost

                    if (
                        verbose and iteration < 2 and candidate_idx < 5
                    ):
                        print(
                            f"  Evaluating candidate ({candidate.x:.1f}, {candidate.y:.1f}), cost {candidate.cost:.1f} -> new_coverage {new_coverage:.1f}, score: {score:.1f}"
                        )

                    if score > best_score:
                        best_score = score
                        best_station = candidate

            if best_station is not None and best_score > 0:
                self.selected_stations.append(best_station)
                self.update_coverage(best_station)

                if verbose:
                    print(
                        f"  Selected: Location ({best_station.x:.1f}, {best_station.y:.1f}) "
                        f"- Score: {best_score:.1f}, Total Covered Demands: {len(self.covered_demands)}\n"
                    )
            else:
                if verbose:
                    print("  No beneficial locations remaining or K stations placed.\n")
                break

        execution_time = time.time() - start_time
        total_coverage = 0
        if self.demands:
            total_coverage = len(self.covered_demands) / len(self.demands) * 100

        if verbose:
            print("Final Results:")
            selected_coords = [
                (round(s.x, 1), round(s.y, 1)) for s in self.selected_stations
            ]
            print(f"- Selected stations: {selected_coords}")

            total_demand_points = (
                len(self.demands) if self.demands else 1
            )
            coverage_percentage = (
                (len(self.covered_demands) / total_demand_points * 100)
                if total_demand_points > 0
                else 0
            )
            print(
                f"- Total coverage: {len(self.covered_demands)}/{len(self.demands)} "
                f"demand points ({coverage_percentage:.1f}%)"
            )
            total_station_cost = sum(station.cost for station in self.selected_stations)
            print(f"- Total cost of selected stations: {total_station_cost:.1f} units")

            total_benefit_value = sum(
                self.demands[i].weight for i in self.covered_demands
            )
            net_benefit = total_benefit_value - total_station_cost
            print(
                f"- Total weighted benefit from covered demands: {total_benefit_value:.1f}"
            )
            print(
                f"- Net benefit (weighted benefit - total cost): {net_benefit:.1f} units"
            )
            print(f"- Execution time: {execution_time:.3f} seconds\n")

        return self.selected_stations

    def visualize_solution(self, title_suffix=""):
        """Create a visualization of the placement solution."""
        if not self.demands and not self.candidates:
            print("No data to visualize.")
            return

        plt.figure(figsize=(10, 8))

        if self.demands:
            demand_x = [d.x for d in self.demands]
            demand_y = [d.y for d in self.demands]
            demand_weights = [d.weight for d in self.demands]
            plt.scatter(
                demand_x,
                demand_y,
                c="blue",
                s=np.array(demand_weights) * 20 + 10,
                alpha=0.6,
                label="Demand Points (size by weight)",
            )

        if self.candidates:
            candidate_x = [c.x for c in self.candidates]
            candidate_y = [c.y for c in self.candidates]
            candidate_costs = [
                c.cost for c in self.candidates
            ]
            plt.scatter(
                candidate_x,
                candidate_y,
                c="lightgray",
                marker="s",
                s=30,
                alpha=0.5,
                label="Candidate Locations",
            )

        plotted_selected_label = False
        for station in self.selected_stations:
            station_label = ""
            if not plotted_selected_label:
                station_label = "Selected Station"
                plotted_selected_label = True

            plt.scatter(
                station.x,
                station.y,
                c="red",
                marker="P",
                s=150,
                edgecolors="black",
                linewidth=1.5,
                label=station_label,
                zorder=5,
            )

            circle = plt.Circle(
                (station.x, station.y),
                self.service_radius,
                color="red",
                fill=True,
                alpha=0.15,
                linestyle="--",
                linewidth=1,
                zorder=3,
            )
            plt.gca().add_patch(circle)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"EV Charging Station Placement{title_suffix}")
        plt.legend(
            loc="upper right", bbox_to_anchor=(1.25, 1)
        )
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


def generate_uniform_random_instance(
    n_candidates: int,
    n_demands: int,
    area_size: Tuple[int, int] = (10, 10),
    seed: int = 42,
) -> Tuple[List[Location], List[Location]]:
    """Generate a random test instance with uniform distribution."""
    if seed is not None:
        np.random.seed(seed)

    candidates = []
    for _ in range(n_candidates):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        cost = np.random.uniform(5, 15)
        candidates.append(Location(x, y, cost=cost))

    demands = []
    for _ in range(n_demands):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        weight = np.random.uniform(1, 5)
        demands.append(Location(x, y, weight=weight))
    return candidates, demands


def generate_clustered_demands_instance(
    n_candidates: int,
    n_demands_per_cluster: int,
    n_clusters: int,
    area_size: Tuple[int, int] = (20, 20),
    cluster_spread: float = 2.0,
    seed: int = 101,
) -> Tuple[List[Location], List[Location]]:
    if seed is not None:
        np.random.seed(seed)

    candidates = []
    for _ in range(n_candidates):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        cost = np.random.uniform(5, 20)
        candidates.append(Location(x, y, cost=cost))

    demands = []
    for _ in range(n_clusters):
        center_x = np.random.uniform(cluster_spread, area_size[0] - cluster_spread)
        center_y = np.random.uniform(cluster_spread, area_size[1] - cluster_spread)
        for _ in range(n_demands_per_cluster):
            x = np.random.normal(center_x, cluster_spread / 2)
            y = np.random.normal(center_y, cluster_spread / 2)
            x = np.clip(x, 0, area_size[0])
            y = np.clip(y, 0, area_size[1])
            weight = np.random.uniform(2, 6)
            demands.append(Location(x, y, weight=weight))
    return candidates, demands


def generate_high_cost_prime_candidates_instance(
    n_candidates: int,
    n_demands: int,
    area_size: Tuple[int, int] = (15, 15),
    seed: int = 202,
) -> Tuple[List[Location], List[Location]]:
    if seed is not None:
        np.random.seed(seed)

    demands = []
    center_area_x, center_area_y = area_size[0] / 2, area_size[1] / 2
    for _ in range(n_demands):
        x = np.random.normal(center_area_x, area_size[0] / 4)
        y = np.random.normal(center_area_y, area_size[1] / 4)
        x = np.clip(x, 0, area_size[0])
        y = np.clip(y, 0, area_size[1])
        weight = np.random.uniform(1, 5)
        demands.append(Location(x, y, weight=weight))

    candidates = []
    num_prime_candidates = n_candidates // 3
    for i in range(n_candidates):
        if i < num_prime_candidates:
            x = np.random.normal(center_area_x, area_size[0] / 5)
            y = np.random.normal(center_area_y, area_size[1] / 5)
            cost = np.random.uniform(25, 50)
        else:
            x = np.random.uniform(0, area_size[0])
            y = np.random.uniform(0, area_size[1])
            cost = np.random.uniform(5, 15)
        x = np.clip(x, 0, area_size[0])
        y = np.clip(y, 0, area_size[1])
        candidates.append(Location(x, y, cost=cost))
    return candidates, demands


def generate_limited_candidates_instance(
    n_candidates: int,
    n_demands: int,
    area_size: Tuple[int, int] = (10, 10),
    seed: int = 303,
) -> Tuple[List[Location], List[Location]]:
    if seed is not None:
        np.random.seed(seed)
    candidates = []
    for _ in range(n_candidates):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        cost = np.random.uniform(8, 12)
        candidates.append(Location(x, y, cost=cost))

    demands = []
    for _ in range(n_demands):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        weight = np.random.uniform(1, 3)
        demands.append(Location(x, y, weight=weight))
    return candidates, demands


def run_and_visualize_instance(
    generator_func, k_stations, service_radius, title_suffix, *args, **kwargs
):
    print(f"\n--- Running Instance: {title_suffix} ---")
    candidates, demands = generator_func(*args, **kwargs)
    placer = ChargingStationPlacer(candidates, demands, service_radius=service_radius)
    placer.greedy_placement(k=k_stations, verbose=True)
    placer.visualize_solution(title_suffix=f" ({title_suffix})")
    return placer

    
def run_performance_test():
    """Run performance tests on different problem sizes using uniform random instances."""
    problem_configs = [
        {
            "name": "Small Uniform",
            "gen_func": generate_uniform_random_instance,
            "args": (20, 30),
            "k": 3,
            "radius": 2.0,
            "seed": 42,
        },
        {
            "name": "Medium Uniform",
            "gen_func": generate_uniform_random_instance,
            "args": (50, 100),
            "k": 7,
            "radius": 2.5,
            "seed": 43,
        },
        {
            "name": "Large Uniform",
            "gen_func": generate_uniform_random_instance,
            "args": (100, 300),
            "k": 10,
            "radius": 3.0,
            "seed": 44,
        },
        {
            "name": "Medium Clustered Demands",
            "gen_func": generate_clustered_demands_instance,
            "args": (60, 25, 3),
            "k": 5,
            "radius": 3.0,
            "seed": 101,
        },
        {
            "name": "Medium High Cost Prime",
            "gen_func": generate_high_cost_prime_candidates_instance,
            "args": (45, 80),
            "k": 6,
            "radius": 2.5,
            "seed": 202,
        },
    ]

    print("Performance Test Results:")
    print("-" * 70)
    print(
        f"{'Test Name':<30} | {'Time (s)':<10} | {'Coverage (%)':<15} | {'Net Benefit':<10}"
    )
    print("-" * 70)

    for config in problem_configs:
        candidates, demands = config["gen_func"](
            *config["args"], seed=config.get("seed")
        )
        placer = ChargingStationPlacer(
            candidates, demands, service_radius=config["radius"]
        )

        start_time = time.time()
        placer.greedy_placement(
            config["k"], verbose=False
        )
        execution_time = time.time() - start_time

        total_demand_points = len(placer.demands) if placer.demands else 1
        coverage_rate = (
            (len(placer.covered_demands) / total_demand_points * 100)
            if total_demand_points > 0
            else 0
        )
        total_station_cost = sum(station.cost for station in placer.selected_stations)
        total_benefit_value = sum(
            placer.demands[i].weight for i in placer.covered_demands
        )
        net_benefit = total_benefit_value - total_station_cost

        print(
            f"{config['name']:<30} | {execution_time:<10.3f} | {coverage_rate:<15.1f} | {net_benefit:<10.1f}"
        )
    print("-" * 70)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Instance 1: Uniform Random
    run_and_visualize_instance(
        generate_uniform_random_instance,
        k_stations=3,
        service_radius=2.5,
        title_suffix="Uniform Random",
        n_candidates=25,
        n_demands=40,
        area_size=(10, 10),
        seed=42,
    )

    # Instance 2: Clustered Demands
    run_and_visualize_instance(
        generate_clustered_demands_instance,
        k_stations=4,
        service_radius=3.0,
        title_suffix="Clustered Demands",
        n_candidates=50,
        n_demands_per_cluster=20,
        n_clusters=3,
        area_size=(20, 20),
        cluster_spread=2.5,
        seed=101,
    )

    # Instance 3: High Cost Prime Candidates
    run_and_visualize_instance(
        generate_high_cost_prime_candidates_instance,
        k_stations=5,
        service_radius=3.0,
        title_suffix="High Cost Prime Candidates",
        n_candidates=40,
        n_demands=60,
        area_size=(15, 15),
        seed=202,
    )

    # Instance 4: Limited Candidates, Many Demands
    run_and_visualize_instance(
        generate_limited_candidates_instance,
        k_stations=3,
        service_radius=2.0,
        title_suffix="Limited Candidates",
        n_candidates=10,
        n_demands=50,
        area_size=(10, 10),
        seed=303,
    )

    # Instance 5: Varying Service Radius (using the same generated instance)
    print("\n--- Running Instance: Varying Service Radius (Clustered Demands data) ---")
    candidates_var_rad, demands_var_rad = generate_clustered_demands_instance(
        n_candidates=50,
        n_demands_per_cluster=15,
        n_clusters=2,
        area_size=(15, 15),
        cluster_spread=2.0,
        seed=404,
    )

    placer_low_radius = ChargingStationPlacer(
        candidates_var_rad, demands_var_rad, service_radius=1.5
    )
    placer_low_radius.greedy_placement(k=3, verbose=True)
    placer_low_radius.visualize_solution(title_suffix=" (Clustered, Low Radius R=1.5)")

    placer_high_radius = ChargingStationPlacer(
        candidates_var_rad, demands_var_rad, service_radius=3.5
    )
    placer_high_radius.greedy_placement(k=3, verbose=True)
    placer_high_radius.visualize_solution(
        title_suffix=" (Clustered, High Radius R=3.5)"
    )

    # Run consolidated performance tests
    print("\n" + "=" * 60 + "\n")
    run_performance_test()
