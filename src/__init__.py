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

    def distance_to(self, other: 'Location') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Location({self.x}, {self.y})"


class ChargingStationPlacer:
    def __init__(self, candidates: List[Location], demands: List[Location],
                 service_radius: float = 2.0):
        self.candidates = candidates
        self.demands = demands
        self.service_radius = service_radius
        self.selected_stations = []
        self.covered_demands = set()

    def calculate_new_coverage(self, station: Location) -> float:
        """Calculate additional weighted coverage provided by a new station."""
        new_coverage = 0.0
        for i, demand in enumerate(self.demands):
            if (i not in self.covered_demands and
                    station.distance_to(demand) <= self.service_radius):
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
            best_score = float('-inf')

            if verbose:
                print(f"Iteration {iteration + 1}:")

            # Evaluate all remaining candidates
            for candidate in self.candidates:
                if candidate not in self.selected_stations:
                    new_coverage = self.calculate_new_coverage(candidate)
                    score = new_coverage - candidate.cost

                    if verbose:
                        print(f"Evaluating location ({candidate.x}, {candidate.y})... "
                              f"Score: {score:.1f}")

                    if score > best_score:
                        best_score = score
                        best_station = candidate

            # Place the best station if beneficial
            if best_station is not None and best_score > 0:
                self.selected_stations.append(best_station)
                self.update_coverage(best_station)

                if verbose:
                    print(f"Selected: Location ({best_station.x}, {best_station.y}) "
                          f"- Score: {best_score:.1f}\n")
            else:
                if verbose:
                    print("No beneficial locations remaining.\n")
                break

        execution_time = time.time() - start_time

        # Calculate final statistics
        total_coverage = len(self.covered_demands)
        total_cost = sum(station.cost for station in self.selected_stations)
        total_benefit = sum(self.demands[i].weight for i in self.covered_demands)
        net_benefit = total_benefit - total_cost

        if verbose:
            print("Final Results:")
            print(f"- Selected stations: {[(s.x, s.y) for s in self.selected_stations]}")
            print(f"- Total coverage: {total_coverage}/{len(self.demands)} "
                  f"demand points ({100 * total_coverage / len(self.demands):.1f}%)")
            print(f"- Total cost: {total_cost:.1f} units")
            print(f"- Net benefit: {net_benefit:.1f} units")
            print(f"- Execution time: {execution_time:.3f} seconds")

        return self.selected_stations

    def visualize_solution(self):
        """Create a visualization of the placement solution."""
        plt.figure(figsize=(10, 8))

        # Plot demand points
        demand_x = [d.x for d in self.demands]
        demand_y = [d.y for d in self.demands]
        demand_weights = [d.weight for d in self.demands]
        plt.scatter(demand_x, demand_y, c='blue', s=np.array(demand_weights) * 20,
                    alpha=0.6, label='Demand Points')

        # Plot candidate locations
        candidate_x = [c.x for c in self.candidates]
        candidate_y = [c.y for c in self.candidates]
        plt.scatter(candidate_x, candidate_y, c='lightgray', marker='s',
                    s=50, alpha=0.5, label='Candidate Locations')

        # Plot selected stations and their coverage
        for station in self.selected_stations:
            # Station location
            plt.scatter(station.x, station.y, c='red', marker='s', s=100,
                        edgecolors='black', linewidth=2, label='Selected Station'
                if station == self.selected_stations[0] else "")

            # Coverage circle
            circle = plt.Circle((station.x, station.y), self.service_radius,
                                color='red', fill=False, alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('EV Charging Station Placement Solution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def generate_test_instance(n_candidates: int, n_demands: int,
                           area_size: Tuple[int, int] = (10, 10)) -> Tuple[List[Location], List[Location]]:
    """Generate a random test instance."""
    np.random.seed(42)  # For reproducible results

    # Generate candidate locations
    candidates = []
    for _ in range(n_candidates):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        cost = np.random.uniform(5, 15)  # Random installation cost
        candidates.append(Location(x, y, cost))

    # Generate demand points
    demands = []
    for _ in range(n_demands):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        weight = np.random.uniform(1, 5)  # Random demand weight
        demands.append(Location(x, y, weight=weight))

    return candidates, demands


def run_performance_test():
    """Run performance tests on different problem sizes."""
    problem_sizes = [
        (10, 15, 3),  # Small
        (50, 100, 10),  # Medium
        (200, 500, 25)  # Large
    ]

    print("Performance Test Results:")
    print("-" * 50)

    for n_candidates, n_demands, k in problem_sizes:
        candidates, demands = generate_test_instance(n_candidates, n_demands)
        placer = ChargingStationPlacer(candidates, demands, service_radius=2.0)

        start_time = time.time()
        placer.greedy_placement(k, verbose=False)
        execution_time = time.time() - start_time

        coverage_rate = len(placer.covered_demands) / len(demands) * 100

        print(f"Size ({n_candidates}, {n_demands}, {k}): "
              f"{execution_time:.3f}s, Coverage: {coverage_rate:.1f}%")


# Main execution
if __name__ == "__main__":
    # Generate test instance
    candidates, demands = generate_test_instance(10, 15)

    # Create placer and solve
    placer = ChargingStationPlacer(candidates, demands, service_radius=2.0)
    solution = placer.greedy_placement(k=3)

    # Visualize solution
    placer.visualize_solution()

    # Run performance tests
    print("\n" + "=" * 60 + "\n")
    run_performance_test()
